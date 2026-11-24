from __future__ import annotations

import re
from http.client import HTTPException
from pathlib import Path
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import cv2

MAX_URLS = 100
MAX_IMAGE_BYTES = 50 * 1024 * 1024
MAX_BATCH_BYTES = 500 * 1024 * 1024
DOWNLOAD_TIMEOUT = 30
DOWNLOAD_DEADLINE = 120
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
MIME_EXTENSIONS = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp",
    "image/bmp": ".bmp", "image/tiff": ".tiff",
}


class ImageDownloadError(ValueError):
    """A download failure that can be shown alongside other batch failures."""


def parse_image_urls(raw_value: str) -> list[str]:
    urls = [line.strip() for line in raw_value.splitlines() if line.strip()]
    if not urls:
        raise ImageDownloadError("请至少输入一个图片下载 URL，每行一个链接。")
    if len(urls) > MAX_URLS:
        raise ImageDownloadError(f"每批最多支持 {MAX_URLS} 个 URL，请分批处理。")
    return urls


def _validate_url(url: str) -> None:
    try:
        parsed = urlsplit(url)
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not parsed.hostname
            or any(character.isspace() or ord(character) < 32 for character in url)
        ):
            raise ValueError
        # Accessing port also validates malformed port numbers.
        parsed.port
    except ValueError as exc:
        raise ImageDownloadError("请输入有效的 HTTP/HTTPS 图片直链，每行一个，不要添加分隔符。") from exc


class _HTTPOnlyRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_url(newurl)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            _add_site_headers(redirected)
        return redirected


def _add_site_headers(req: Request) -> None:
    hostname = (urlsplit(req.full_url).hostname or "").lower()
    if hostname == "pximg.net" or hostname.endswith(".pximg.net"):
        # Pixiv's image CDN requires its site as the source of image requests.
        # Recompute on redirects so this header never follows unrelated hosts.
        req.add_unredirected_header("Referer", "https://www.pixiv.net/")


def _image_request(url: str) -> Request:
    req = Request(url, headers={"User-Agent": BROWSER_USER_AGENT, "Accept": "image/*"})
    _add_site_headers(req)
    return req


def _download_name(response) -> str:
    name = response.headers.get_filename() or unquote(urlsplit(response.url).path)
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r'[\x00-\x1f\x7f<>:"/\\|?*]', "_", name).strip(" .")
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        suffix = MIME_EXTENSIONS.get(response.headers.get_content_type(), ".png")
        stem = Path(name).stem if name else "download"
    else:
        stem = Path(name).stem
    # Leave room for the upscale suffix and filesystem filename limits.
    stem = stem.encode("utf-8")[:180].decode("utf-8", errors="ignore") or "download"
    return f"{stem}{suffix}"


def download_image(url: str, directory: Path, max_bytes: int = MAX_IMAGE_BYTES) -> Path:
    _validate_url(url)
    limit = min(max_bytes, MAX_IMAGE_BYTES)
    if limit <= 0:
        raise ImageDownloadError("本批下载已达到 500 MB 上限，请分批处理。")
    target: Path | None = None
    completed = False
    started = monotonic()
    try:
        # Use this machine's network and urllib's environment/system proxy discovery.
        opener = build_opener(_HTTPOnlyRedirectHandler())
        req = _image_request(url)
        with opener.open(req, timeout=DOWNLOAD_TIMEOUT) as response:
            _validate_url(response.url)
            length = response.headers.get("Content-Length", "")
            if length.isdigit() and int(length) > limit:
                raise ImageDownloadError("图片超过下载大小限制（单张 50 MB，每批 500 MB）。")
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / _download_name(response)
            size = 0
            with target.open("wb") as output:
                while True:
                    if monotonic() - started > DOWNLOAD_DEADLINE:
                        raise ImageDownloadError("图片下载超时，请稍后重试。")
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > limit:
                        raise ImageDownloadError("图片超过下载大小限制（单张 50 MB，每批 500 MB）。")
                    output.write(chunk)
            if length.isdigit() and size != int(length):
                raise ImageDownloadError("图片下载不完整，请稍后重试。")
            if not size or not cv2.haveImageReader(str(target)):
                raise ImageDownloadError("下载内容不是可读取的图片，请使用图片文件直链。")
            completed = True
            return target
    except HTTPError as exc:
        exc.close()
        if exc.code == 403:
            raise ImageDownloadError("下载失败：HTTP 403，图片站拒绝访问，请检查链接是否需要登录或额外访问权限。") from exc
        raise ImageDownloadError(f"下载失败：HTTP {exc.code}。") from exc
    except (URLError, TimeoutError, OSError, ValueError, HTTPException) as exc:
        if isinstance(exc, ImageDownloadError):
            raise
        raise ImageDownloadError("下载失败，请检查链接、网络连接及访问权限。") from exc
    finally:
        if not completed and target is not None:
            target.unlink(missing_ok=True)
