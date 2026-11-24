from __future__ import annotations

import os
import shutil
import sys
import types
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock


def _positive_env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except ValueError:
        return default


CPU_THREAD_COUNT = _positive_env_int("REAL_ESRGAN_CPU_THREADS", max(1, os.cpu_count() or 1))
for env_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(env_name, str(CPU_THREAD_COUNT))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch

try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ModuleNotFoundError:
    from torchvision.transforms import functional as tv_functional

    compatibility_module = types.ModuleType("torchvision.transforms.functional_tensor")
    compatibility_module.rgb_to_grayscale = tv_functional.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = compatibility_module

from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

MODEL_NAME = "realesr-general-x4v3"
LOW_PROCESS_DENOISE_CAP = 2
LOW_PROCESS_MODEL_WEIGHT = 0.25

GENERAL_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
    "realesr-general-x4v3.pth"
)
WDN_MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
    "realesr-general-wdn-x4v3.pth"
)


class UpscaleRequestError(RuntimeError):
    """User-facing exception for predictable upscale failures."""


@dataclass(slots=True)
class BatchImage:
    source_path: Path
    input_name: str
    output_subdir: Path = Path()
    archive_subdir: Path | None = None


@dataclass(slots=True)
class UpscaleResult:
    input_name: str
    output_name: str
    output_path: str
    output_relative_path: str
    archive_path: str | None
    scale: int
    denoise: float
    denoise_level: int
    denoise_label: str
    width: int
    height: int
    processing_mode: str = "standard"

    def to_dict(self, preview_url: str) -> dict[str, object]:
        return {
            "input_name": self.input_name,
            "output_name": self.output_name,
            "output_path": self.output_path,
            "output_relative_path": self.output_relative_path,
            "archive_path": self.archive_path,
            "scale": self.scale,
            "denoise": self.denoise,
            "denoise_level": self.denoise_level,
            "denoise_label": self.denoise_label,
            "processing_mode": self.processing_mode,
            "model_name": MODEL_NAME,
            "width": self.width,
            "height": self.height,
            "preview_url": preview_url,
        }


class RealESRGANBatchService:
    def __init__(
        self,
        output_dir: Path,
        weights_dir: Path,
        archive_dir: Path,
        tile: int = 0,
    ) -> None:
        self.output_dir = output_dir
        self.weights_dir = weights_dir
        self.archive_dir = archive_dir
        self.input_dir = archive_dir.parent
        self.tile = tile
        # RealESRGANer stores mutable input/output tensors; never share concurrent inference.
        self.batch_lock = RLock()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.cpu_threads = CPU_THREAD_COUNT
        self.device = self._select_device()
        self.runtime_profile = self._configure_runtime()
        self.last_device = self.device
        self.last_runtime_profile = self.runtime_profile
        self.last_warnings: list[dict[str, str]] = []

    def process_batch(
        self,
        files: list[BatchImage],
        scale: int,
        denoise_level: int,
        device_mode: str = "auto",
        processing_mode: str = "standard",
    ) -> tuple[list[UpscaleResult], list[dict[str, str]]]:
        if processing_mode not in {"standard", "low"}:
            raise UpscaleRequestError("不支持的处理模式。")
        if scale not in {2, 3, 4} or not 0 <= denoise_level <= 10:
            raise UpscaleRequestError("倍率或降噪等级无效。")
        with self.batch_lock:
            return self._process_batch(files, scale, denoise_level, device_mode, processing_mode)

    @staticmethod
    def effective_denoise_level(denoise_level: int, processing_mode: str) -> int:
        return min(denoise_level, LOW_PROCESS_DENOISE_CAP) if processing_mode == "low" else denoise_level

    @staticmethod
    def _read_image(item: BatchImage) -> np.ndarray:
        image = cv2.imread(str(item.source_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise UpscaleRequestError("无法读取图片内容。")
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def _process_batch(
        self,
        files: list[BatchImage],
        scale: int,
        denoise_level: int,
        device_mode: str,
        processing_mode: str,
    ) -> tuple[list[UpscaleResult], list[dict[str, str]]]:
        denoise_level = self.effective_denoise_level(denoise_level, processing_mode)
        denoise_strength = self._denoise_strength(denoise_level)
        results: list[UpscaleResult] = []
        failures: list[dict[str, str]] = []
        warnings: list[dict[str, str]] = []
        requested_device_type, request_warning = self._resolve_device_type(device_mode)
        mps_disabled = False
        fallback_used = False
        active_device_type = requested_device_type
        if request_warning:
            warnings.append({"input_name": "设备", "message": request_warning})

        def collect_write(pending: tuple[BatchImage, Future[UpscaleResult]] | None) -> None:
            if pending is None:
                return
            item, future = pending
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"input_name": item.input_name, "error": str(exc)})

        # Bounded pipeline: one prefetched input and one pending output, regardless of batch size.
        # Torch owns the CPU compute threads; I/O workers do not run model inference.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="esr-read") as reader, \
             ThreadPoolExecutor(max_workers=1, thread_name_prefix="esr-write") as writer:
            read_future = reader.submit(self._read_image, files[0]) if files else None
            pending_write = None
            for index, item in enumerate(files):
                current_read = read_future
                read_future = reader.submit(self._read_image, files[index + 1]) if index + 1 < len(files) else None
                try:
                    image = current_read.result()
                    output_image, used_device_type, warning_message, mps_disabled = self._enhance_image(
                        image=image,
                        scale=scale,
                        denoise_strength=denoise_strength,
                        requested_device_type=requested_device_type,
                        mps_disabled=mps_disabled,
                    )
                    active_device_type = used_device_type
                    if warning_message:
                        fallback_used = True
                        warnings.append({"input_name": item.input_name, "message": warning_message})
                    if processing_mode == "low":
                        output_image = self._blend_low_processing(image, output_image)
                    collect_write(pending_write)
                    pending_write = None
                    pending_write = (item, writer.submit(
                        self._save_result, item, output_image, scale, denoise_level, processing_mode,
                    ))
                    del image, output_image
                except Exception as exc:
                    failures.append({"input_name": item.input_name, "error": str(exc)})
            collect_write(pending_write)

        # Preserve the allocator across images; release unused CUDA blocks once per batch.
        self._flush_device_cache(active_device_type)
        self.last_device = torch.device("cpu" if fallback_used else active_device_type)
        self.last_runtime_profile = "metal-fallback" if fallback_used else self._runtime_label(active_device_type)
        self.last_warnings = warnings
        return results, failures

    @staticmethod
    def _blend_low_processing(image: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
        height, width = enhanced.shape[:2]
        base = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
        # Blend in strips to avoid an additional full-size floating point image.
        output = np.empty_like(base)
        for top in range(0, height, 256):
            bottom = min(top + 256, height)
            output[top:bottom] = cv2.addWeighted(
                base[top:bottom], 1 - LOW_PROCESS_MODEL_WEIGHT,
                enhanced[top:bottom], LOW_PROCESS_MODEL_WEIGHT, 0,
                dtype=cv2.CV_16U if base.dtype == np.uint16 else cv2.CV_8U,
            )
        if image.ndim == 3 and image.shape[2] == 4:
            output[..., 3] = base[..., 3]
        return output

    def _save_result(
        self,
        item: BatchImage,
        output_image: np.ndarray,
        scale: int,
        denoise_level: int,
        processing_mode: str,
    ) -> UpscaleResult:
        output_path = self._build_output_path(
            original_name=item.source_path.name, output_subdir=item.output_subdir,
            scale=scale, denoise_level=denoise_level, processing_mode=processing_mode,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), output_image):
            raise UpscaleRequestError("输出图片写入失败。")
        archive_path = None
        if item.archive_subdir is not None:
            archive_path = self._archive_original(item.source_path, item.archive_subdir)
        height, width = output_image.shape[:2]
        return UpscaleResult(
            input_name=item.input_name, output_name=output_path.name,
            output_path=str(output_path),
            output_relative_path=output_path.relative_to(self.output_dir).as_posix(),
            archive_path=str(archive_path) if archive_path else None,
            scale=scale, denoise=self._denoise_strength(denoise_level),
            denoise_level=denoise_level, denoise_label=self._denoise_label(denoise_level),
            width=width, height=height, processing_mode=processing_mode,
        )

    def _select_device(self) -> torch.device:
        preferred_device = os.environ.get("REAL_ESRGAN_DEVICE", "").strip().lower()
        if preferred_device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if preferred_device == "mps":
            if self._mps_available():
                return torch.device("mps")
            return torch.device("cpu")
        if preferred_device == "cpu":
            return torch.device("cpu")

        if torch.cuda.is_available():
            return torch.device("cuda")
        if self._mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _configure_runtime(self) -> str:
        cv2.setUseOptimized(True)
        # Avoid nesting an all-core OpenCV pool inside the Torch/I/O pipeline.
        cv2.setNumThreads(1)
        torch.set_num_threads(self.cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        if self.device.type == "mps":
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            return "metal"
        if self.device.type == "cpu":
            return "cpu-multicore"
        return self.device.type

    def _runtime_label(self, device_type: str) -> str:
        if device_type == "mps":
            return "metal"
        if device_type == "cpu":
            return "cpu-multicore"
        return device_type

    def _mps_available(self) -> bool:
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    def available_devices(self) -> dict[str, bool]:
        return {
            "cpu": True,
            "cuda": torch.cuda.is_available(),
            "metal": self._mps_available(),
        }

    def _resolve_device_type(self, device_mode: str) -> tuple[str, str | None]:
        if device_mode == "cpu":
            return "cpu", None
        if device_mode == "metal":
            if self._mps_available():
                return "mps", None
            return "cpu", "Metal 当前不可用，已切换到 CPU。"
        if device_mode == "cuda":
            if torch.cuda.is_available():
                return "cuda", None
            return "cpu", "CUDA 当前不可用，已切换到 CPU。"
        if torch.cuda.is_available():
            return "cuda", None
        if self._mps_available():
            return "mps", None
        return "cpu", None

    def _flush_device_cache(self, active_device_type: str) -> None:
        # torch.mps.empty_cache() has caused interpreter crashes on macOS 15 / Apple Silicon
        # in this project, so MPS cleanup is intentionally disabled.
        if active_device_type == "cuda":
            torch.cuda.empty_cache()

    def _build_model(self) -> SRVGGNetCompact:
        return SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )

    def _ensure_weights(self) -> tuple[Path, Path]:
        expected_files = {
            "realesr-general-x4v3.pth": GENERAL_MODEL_URL,
            "realesr-general-wdn-x4v3.pth": WDN_MODEL_URL,
        }

        for file_name, url in expected_files.items():
            target = self.weights_dir / file_name
            if target.exists():
                continue
            try:
                load_file_from_url(
                    url=url,
                    model_dir=str(self.weights_dir),
                    progress=True,
                    file_name=file_name,
                )
            except Exception as exc:  # noqa: BLE001
                raise UpscaleRequestError(
                    "模型权重下载失败。请确认网络可用，或手动把官方权重放到 weights 目录。"
                ) from exc

        return (
            self.weights_dir / "realesr-general-x4v3.pth",
            self.weights_dir / "realesr-general-wdn-x4v3.pth",
        )

    @lru_cache(maxsize=3)
    def _get_upsampler(self, denoise_strength: float, device_type: str) -> RealESRGANer:
        general_model_path, wdn_model_path = self._ensure_weights()
        device = torch.device(device_type)
        upsampler_kwargs = {
            "scale": 4,
            "model_path": [str(general_model_path), str(wdn_model_path)],
            "dni_weight": [denoise_strength, 1 - denoise_strength],
            "model": self._build_model(),
            "tile": self.tile,
            "tile_pad": 10,
            "pre_pad": 0,
            "half": device_type == "cuda",
        }

        try:
            return RealESRGANer(device=device, **upsampler_kwargs)
        except TypeError:
            gpu_id = 0 if device_type == "cuda" else None
            return RealESRGANer(gpu_id=gpu_id, **upsampler_kwargs)

    @torch.inference_mode()
    def _enhance_image(
        self,
        image: np.ndarray,
        scale: int,
        denoise_strength: float,
        requested_device_type: str,
        mps_disabled: bool,
    ) -> tuple[np.ndarray, str, str | None, bool]:
        warning_message = None
        last_error: Exception | None = None

        if requested_device_type == "mps":
            device_candidates = ["cpu"] if mps_disabled else ["mps", "cpu"]
        else:
            device_candidates = [requested_device_type]

        for device_type in device_candidates:
            upsampler = None
            try:
                upsampler = self._get_upsampler(round(denoise_strength, 2), device_type)
                tile_size = self._select_outer_tile_size(image.shape[:2], device_type)
                if tile_size > 0:
                    output_image = self._enhance_image_tiled(
                        image=image,
                        upsampler=upsampler,
                        outscale=scale,
                        tile_size=tile_size,
                        tile_pad=24,
                    )
                else:
                    output_image, _ = upsampler.enhance(image, outscale=scale)
                if self._looks_invalid_output(image, output_image):
                    raise UpscaleRequestError("推理结果异常，输出几乎全黑。")
                return output_image, device_type, warning_message, mps_disabled
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if device_type == "mps":
                    mps_disabled = True
                    warning_message = (
                        "Metal 推理结果异常，已自动回退到 CPU 继续处理。"
                    )
                    continue
                break
            finally:
                # Cached helpers must not retain large tensors between images/requests.
                if upsampler is not None:
                    upsampler.img = None
                    upsampler.output = None

        raise UpscaleRequestError(str(last_error or "图片超分失败。"))

    def _select_outer_tile_size(
        self,
        image_shape: tuple[int, int],
        device_type: str,
    ) -> int:
        height, width = image_shape
        pixels = height * width
        max_side = max(height, width)

        if device_type == "cpu":
            if pixels >= 24_000_000 or max_side >= 5000:
                return 1024
            if pixels >= 10_000_000 or max_side >= 3200:
                return 1280
            if pixels >= 4_000_000 or max_side >= 2000:
                return 1536
            return 0

        if device_type == "mps":
            if pixels >= 12_000_000 or max_side >= 4000:
                return 768
            if pixels >= 5_000_000 or max_side >= 2400:
                return 1024
            return 0

        if pixels >= 16_000_000 or max_side >= 4000:
            return 1024
        return 0

    def _enhance_image_tiled(
        self,
        image: np.ndarray,
        upsampler: RealESRGANer,
        outscale: int,
        tile_size: int,
        tile_pad: int,
    ) -> np.ndarray:
        input_height, input_width = image.shape[:2]
        output_image: np.ndarray | None = None

        for top in range(0, input_height, tile_size):
            bottom = min(top + tile_size, input_height)
            for left in range(0, input_width, tile_size):
                right = min(left + tile_size, input_width)

                pad_top = max(top - tile_pad, 0)
                pad_bottom = min(bottom + tile_pad, input_height)
                pad_left = max(left - tile_pad, 0)
                pad_right = min(right + tile_pad, input_width)

                input_tile = image[pad_top:pad_bottom, pad_left:pad_right]
                output_tile, _ = upsampler.enhance(input_tile, outscale=outscale)
                if self._looks_invalid_output(input_tile, output_tile):
                    raise UpscaleRequestError("分块推理结果异常，输出几乎全黑。")

                if output_image is None:
                    output_shape = (
                        input_height * outscale,
                        input_width * outscale,
                    )
                    if output_tile.ndim == 3:
                        output_shape = (*output_shape, output_tile.shape[2])
                    output_image = np.empty(output_shape, dtype=output_tile.dtype)

                crop_top = (top - pad_top) * outscale
                crop_bottom = crop_top + (bottom - top) * outscale
                crop_left = (left - pad_left) * outscale
                crop_right = crop_left + (right - left) * outscale

                output_image[
                    top * outscale:bottom * outscale,
                    left * outscale:right * outscale,
                    ...,
                ] = output_tile[crop_top:crop_bottom, crop_left:crop_right, ...]

        if output_image is None:
            raise UpscaleRequestError("没有生成任何分块输出。")
        return output_image

    def _looks_invalid_output(self, input_image, output_image) -> bool:
        if output_image is None or getattr(output_image, "size", 0) == 0:
            return True

        input_rgb = input_image[..., :3] if input_image.ndim == 3 else input_image
        output_rgb = output_image[..., :3] if output_image.ndim == 3 else output_image

        input_max = float(input_rgb.max())
        input_mean = float(input_rgb.mean())
        output_max = float(output_rgb.max())
        output_mean = float(output_rgb.mean())

        if input_max >= 32 and output_max <= 2:
            return True
        if input_mean >= 8 and output_mean <= 0.5:
            return True
        return False

    def _build_output_path(
        self,
        original_name: str,
        output_subdir: Path,
        scale: int,
        denoise_level: int,
        processing_mode: str = "standard",
    ) -> Path:
        original_path = Path(original_name)
        suffix = original_path.suffix.lower() or ".png"
        stem = original_path.stem
        parent_dir = self.output_dir / output_subdir
        denoise_tag = self._denoise_label(denoise_level)
        if processing_mode == "low":
            denoise_tag += "-low"

        candidate = parent_dir / f"{stem}-{scale}x-{denoise_tag}{suffix}"
        counter = 1
        while candidate.exists():
            candidate = parent_dir / f"{stem}-{scale}x-{denoise_tag}-{counter}{suffix}"
            counter += 1
        return candidate

    def _archive_original(self, source_path: Path, archive_subdir: Path) -> Path:
        archive_parent = self.archive_dir / archive_subdir
        archive_parent.mkdir(parents=True, exist_ok=True)

        candidate = archive_parent / source_path.name
        counter = 1
        while candidate.exists():
            candidate = archive_parent / (
                f"{source_path.stem}_{counter}{source_path.suffix.lower()}"
            )
            counter += 1

        shutil.move(str(source_path), str(candidate))
        self._cleanup_empty_input_dirs(source_path.parent)
        return candidate

    def _cleanup_empty_input_dirs(self, directory: Path) -> None:
        current = directory
        while current != self.input_dir and current.exists():
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent

    def _denoise_strength(self, denoise_level: int) -> float:
        return round(denoise_level / 10, 1)

    def _denoise_label(self, denoise_level: int) -> str:
        return f"d{denoise_level}x"
