from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for

from app.image_download import (
    ALLOWED_EXTENSIONS, MAX_BATCH_BYTES, ImageDownloadError, download_image, parse_image_urls,
)
from app.realesrgan_service import (
    MODEL_NAME, BatchImage, RealESRGANBatchService, UpscaleRequestError,
)

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "Images_Input"
ARCHIVE_DIR = INPUT_DIR / "Archive"
OUTPUT_DIR = BASE_DIR / "Images_Output"
UPLOAD_DIR = BASE_DIR / "uploads"
WEIGHTS_DIR = BASE_DIR / "weights"

for folder in (INPUT_DIR, ARCHIVE_DIR, OUTPUT_DIR, UPLOAD_DIR, WEIGHTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

service = RealESRGANBatchService(
    output_dir=OUTPUT_DIR,
    weights_dir=WEIGHTS_DIR,
    archive_dir=ARCHIVE_DIR,
)


def _parse_scale(raw_value: str) -> int:
    try:
        scale = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise UpscaleRequestError("倍率必须是 2、3 或 4。") from exc

    if scale not in {2, 3, 4}:
        raise UpscaleRequestError("倍率必须是 2、3 或 4。")
    return scale


def _parse_source_mode(raw_value: str | None) -> str:
    source_mode = raw_value or "upload"
    if source_mode not in {"upload", "input-folder", "url"}:
        raise UpscaleRequestError("不支持的处理来源。")
    return source_mode


def _parse_processing_mode(raw_value: str | None) -> str:
    mode = raw_value or "standard"
    if mode not in {"standard", "low"}:
        raise UpscaleRequestError("不支持的处理模式。")
    return mode


def _parse_device_mode(raw_value: str | None) -> str:
    device_mode = raw_value or "auto"
    if device_mode not in {"auto", "cpu", "metal", "cuda"}:
        raise UpscaleRequestError("不支持的计算设备。")
    return device_mode


def _parse_denoise_level(raw_value: str | None, legacy_value: str | None) -> int:
    if raw_value is not None:
        try:
            denoise_level = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise UpscaleRequestError("降噪强度必须是 0 到 10 之间的整数。") from exc

        if not 0 <= denoise_level <= 10:
            raise UpscaleRequestError("降噪强度必须在 0 到 10 之间。")
        return denoise_level

    try:
        denoise_strength = float(legacy_value or "0.5")
    except (TypeError, ValueError) as exc:
        raise UpscaleRequestError("降噪强度必须是 0 到 1 之间的数字。") from exc

    if not 0.0 <= denoise_strength <= 1.0:
        raise UpscaleRequestError("降噪强度必须在 0 到 1 之间。")
    return int(round(denoise_strength * 10))


def _scan_input_folder() -> list[BatchImage]:
    items: list[BatchImage] = []

    for root, dirs, files in os.walk(INPUT_DIR):
        dirs[:] = sorted(directory for directory in dirs if directory != "Archive")
        root_path = Path(root)
        relative_parent = root_path.relative_to(INPUT_DIR)

        for file_name in sorted(files, key=str.lower):
            source_path = root_path / file_name
            if source_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue

            relative_path = (
                relative_parent / file_name if relative_parent != Path(".") else Path(file_name)
            )
            items.append(
                BatchImage(
                    source_path=source_path,
                    input_name=relative_path.as_posix(),
                    output_subdir=relative_parent if relative_parent != Path(".") else Path(),
                    archive_subdir=relative_parent if relative_parent != Path(".") else Path(),
                )
            )

    return items


@app.get("/")
def index() -> str:
    return render_template(
        "index.html",
        input_dir=str(INPUT_DIR),
        archive_dir=str(ARCHIVE_DIR),
        output_dir=str(OUTPUT_DIR),
    )


@app.get("/outputs/<path:filename>")
def serve_output(filename: str):
    return send_from_directory(OUTPUT_DIR, filename)


@app.get("/api/input-images")
def list_input_images():
    items = _scan_input_folder()
    return jsonify(
        {
            "input_dir": str(INPUT_DIR),
            "archive_dir": str(ARCHIVE_DIR),
            "count": len(items),
            "files": [item.input_name for item in items],
        }
    )


@app.get("/api/runtime")
def runtime_info():
    return jsonify(
        {
            "model_name": MODEL_NAME,
            "available_devices": service.available_devices(),
            "default_device": service.device.type,
            "runtime_profile": service.runtime_profile,
            "cpu_threads": service.cpu_threads,
        }
    )


@app.post("/api/upscale")
def upscale():
    # Keep folder scans, inference, archiving and response metadata in one transaction.
    with service.batch_lock:
        return _upscale()


def _upscale():
    try:
        source_mode = _parse_source_mode(request.form.get("source_mode"))
        device_mode = _parse_device_mode(request.form.get("device_mode"))
        processing_mode = _parse_processing_mode(request.form.get("processing_mode"))
        scale = _parse_scale(request.form.get("scale", "4"))
        denoise_level = _parse_denoise_level(
            request.form.get("denoise_level"),
            request.form.get("denoise"),
        )
        urls = parse_image_urls(request.form.get("urls", "")) if source_mode == "url" else []
    except (UpscaleRequestError, ImageDownloadError) as exc:
        return jsonify({"error": str(exc)}), 400

    upload_items: list[Path] = []
    batch_items: list[BatchImage] = []
    download_failures: list[dict[str, str]] = []
    download_directory = None
    request_prefix = uuid4().hex

    try:
        if source_mode == "input-folder":
            batch_items = _scan_input_folder()
            if not batch_items:
                return jsonify({"error": "Images_Input 中没有可处理的图片。"}), 400
        elif source_mode == "url":
            download_directory = TemporaryDirectory(prefix="url_", dir=UPLOAD_DIR)
            remaining_bytes = MAX_BATCH_BYTES
            for index, url in enumerate(urls, start=1):
                try:
                    image_path = download_image(
                        url, Path(download_directory.name) / str(index), remaining_bytes,
                    )
                    remaining_bytes -= image_path.stat().st_size
                    batch_items.append(BatchImage(source_path=image_path, input_name=image_path.name))
                except ImageDownloadError as exc:
                    download_failures.append({"input_name": url, "error": str(exc)})

            if not batch_items:
                return jsonify({
                    "error": "所有链接均下载失败，请检查下方详情。",
                    "source_mode": source_mode,
                    "results": [],
                    "failures": download_failures,
                }), 502
        else:
            files = request.files.getlist("files")
            if not files:
                return jsonify({"error": "请至少上传一张图片。"}), 400

            for index, storage in enumerate(files, start=1):
                if not storage.filename:
                    continue

                original_name = Path(storage.filename).name
                suffix = Path(original_name).suffix.lower()
                if suffix not in ALLOWED_EXTENSIONS:
                    return jsonify({"error": f"不支持的文件类型：{original_name}"}), 400

                upload_path = UPLOAD_DIR / f"{request_prefix}_{index:03d}{suffix}"
                storage.save(upload_path)
                upload_items.append(upload_path)
                batch_items.append(
                    BatchImage(
                        source_path=upload_path,
                        input_name=original_name,
                    )
                )

            if not batch_items:
                return jsonify({"error": "没有可处理的有效图片。"}), 400

        results, failures = service.process_batch(
            files=batch_items,
            scale=scale,
            denoise_level=denoise_level,
            device_mode=device_mode,
            processing_mode=processing_mode,
        )
        failures = download_failures + failures
    except UpscaleRequestError as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        for upload_path in upload_items:
            upload_path.unlink(missing_ok=True)
        if download_directory is not None:
            download_directory.cleanup()

    response_results = [
        result.to_dict(
            preview_url=url_for("serve_output", filename=result.output_relative_path),
        )
        for result in results
    ]

    effective_level = service.effective_denoise_level(denoise_level, processing_mode)
    payload = {
        "message": f"完成 {len(results)} 张图片处理。",
        "source_mode": source_mode,
        "device_mode": device_mode,
        "processing_mode": processing_mode,
        "model_name": MODEL_NAME,
        "requested_denoise_level": denoise_level,
        "input_dir": str(INPUT_DIR),
        "archive_dir": str(ARCHIVE_DIR),
        "output_dir": str(OUTPUT_DIR),
        "device": service.last_device.type,
        "runtime_profile": service.last_runtime_profile,
        "available_devices": service.available_devices(),
        "cpu_threads": service.cpu_threads,
        "scale": scale,
        "denoise": service._denoise_strength(effective_level),
        "denoise_level": effective_level,
        "denoise_label": service._denoise_label(effective_level),
        "warnings": service.last_warnings,
        "results": response_results,
        "failures": failures,
    }
    if not response_results and failures:
        payload["error"] = failures[0]["error"]
        return jsonify(payload), 500
    return jsonify(payload)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False, threaded=True)
