import importlib.util
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.request import getproxies

import cv2
import numpy as np

from app import image_download
from app.image_download import ImageDownloadError, download_image, parse_image_urls
from app.realesrgan_service import RealESRGANBatchService, UpscaleRequestError


IMAGE = np.full((8, 9, 3), 100, np.uint8)
PNG = cv2.imencode('.png', IMAGE)[1].tobytes()


class ImageHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path in {'/redirect', '/bad-redirect'}:
            self.send_response(302)
            self.send_header('Location', '/photo.png' if self.path == '/redirect' else 'file:///etc/hosts')
            self.end_headers()
            return
        if self.path == '/missing':
            self.send_error(404)
            return
        if self.path == '/forbidden':
            self.send_error(403)
            return
        body = b'<html>not an image</html>' if self.path == '/html' else PNG
        if self.path == '/empty':
            body = b''
        self.send_response(200)
        self.send_header('Content-Type', 'image/png')
        if self.path == '/named':
            self.send_header('Content-Disposition', "attachment; filename*=UTF-8''..%2F..%2F%E5%8E%9F%E5%9B%BE.png")
        if self.path != '/no-length':
            self.send_header('Content-Length', str(len(body) + (100 if self.path == '/truncated' else 0)))
        self.end_headers()
        self.wfile.write(body)


class DownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), ImageHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f'http://127.0.0.1:{cls.server.server_port}'
        spec = importlib.util.spec_from_file_location('url_webapp', Path(__file__).resolve().parents[1] / 'app.py')
        cls.web = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.web)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_only_newlines_separate_urls_and_blank_lines_are_ignored(self):
        first = f'{self.base_url}/photo.png?token=a,b;c&x=1'
        second = f'{self.base_url}/named'
        self.assertEqual(parse_image_urls(f'  {first} \r\n\r\n {second}\n'), [first, second])
        for value in (' \n\r\n', '\n'.join([first] * 101)):
            with self.assertRaises(ImageDownloadError):
                parse_image_urls(value)

    def test_download_redirect_extensionless_and_safe_unicode_filename(self):
        for endpoint, expected in (('/photo.png', 'photo.png'), ('/redirect', 'photo.png'),
                                   ('/download?id=1', 'download.png'), ('/named', '原图.png')):
            with self.subTest(endpoint=endpoint):
                target = download_image(self.base_url + endpoint, self.root)
                self.assertEqual(target.name, expected)
                self.assertEqual(target.parent, self.root)
                np.testing.assert_array_equal(cv2.imread(str(target)), IMAGE)

    def test_pixiv_headers_are_scoped_to_its_image_hosts(self):
        for host in ('i.pximg.net', 'pximg.net', 'S.PXIMG.NET'):
            req = image_download._image_request(f'https://{host}/image.png')
            self.assertEqual(req.get_header('Referer'), 'https://www.pixiv.net/')
            self.assertIn('Mozilla/5.0', req.get_header('User-agent'))
        for host in ('example.com', 'pximg.net.example.com', 'notpximg.net'):
            req = image_download._image_request(f'https://{host}/image.png')
            self.assertIsNone(req.get_header('Referer'))

    def test_redirects_apply_pixiv_headers_only_to_the_target_host(self):
        handler = image_download._HTTPOnlyRedirectHandler()
        original = image_download._image_request('https://example.com/image.png')
        pixiv = handler.redirect_request(original, None, 302, 'Found', {}, 'https://i.pximg.net/image.png')
        self.assertEqual(pixiv.get_header('Referer'), 'https://www.pixiv.net/')
        other = handler.redirect_request(pixiv, None, 302, 'Found', {}, 'https://example.com/image.png')
        self.assertIsNone(other.get_header('Referer'))
        self.assertEqual(other.get_header('User-agent'), original.get_header('User-agent'))

    def test_downloader_preserves_local_environment_proxy_settings(self):
        with patch.dict('os.environ', {'https_proxy': 'http://127.0.0.1:17897'}, clear=True):
            opener = image_download.build_opener(image_download._HTTPOnlyRedirectHandler())
            proxy_handler = next(handler for handler in opener.handlers if hasattr(handler, 'proxies'))
            self.assertEqual(proxy_handler.proxies['https'], getproxies()['https'])
            self.assertEqual(proxy_handler.proxies['https'], 'http://127.0.0.1:17897')

    def test_access_denied_has_a_clear_error_message(self):
        with self.assertRaisesRegex(ImageDownloadError, 'HTTP 403.*拒绝访问'):
            download_image(self.base_url + '/forbidden', self.root)

    def test_invalid_urls_never_open_a_connection(self):
        with patch.object(image_download, 'build_opener') as opener:
            for url in ('file:///tmp/a.png', 'ftp://example.com/a.png', 'https://',
                        'https://example.com:bad/a.png', f'{self.base_url}/a.png {self.base_url}/b.png'):
                with self.subTest(url=url), self.assertRaises(ImageDownloadError):
                    download_image(url, self.root)
            opener.assert_not_called()

    def test_http_error_bad_redirect_and_non_images_are_reported(self):
        for endpoint in ('/missing', '/bad-redirect', '/html', '/empty', '/truncated'):
            with self.subTest(endpoint=endpoint), self.assertRaises(ImageDownloadError):
                download_image(self.base_url + endpoint, self.root)
            self.assertFalse(list(self.root.iterdir()))

    def test_size_limits_with_and_without_content_length_remove_partial_files(self):
        for endpoint in ('/photo.png', '/no-length'):
            with self.subTest(endpoint=endpoint), self.assertRaisesRegex(ImageDownloadError, '大小限制'):
                download_image(self.base_url + endpoint, self.root, max_bytes=10)
            self.assertFalse(list(self.root.iterdir()))
        with patch.object(image_download, 'build_opener') as opener:
            with self.assertRaisesRegex(ImageDownloadError, '上限'):
                download_image(self.base_url + '/photo.png', self.root, max_bytes=0)
            opener.assert_not_called()

    def test_download_timeout_is_reported_and_deadline_cleans_up(self):
        with patch.object(image_download, 'build_opener') as opener:
            opener.return_value.open.side_effect = TimeoutError()
            with self.assertRaises(ImageDownloadError):
                download_image(self.base_url + '/photo.png', self.root)
        with patch.object(image_download, 'monotonic', side_effect=[0, 121]):
            with self.assertRaisesRegex(ImageDownloadError, '超时'):
                download_image(self.base_url + '/photo.png', self.root)
        self.assertFalse(list(self.root.iterdir()))

    @staticmethod
    def enhance(image, scale, denoise_strength, requested_device_type, mps_disabled):
        return cv2.resize(image, None, fx=scale, fy=scale), 'cpu', None, mps_disabled

    def post_urls(self, urls, **fields):
        uploads = self.root / 'uploads'
        uploads.mkdir(exist_ok=True)
        service = RealESRGANBatchService(self.root / 'out', self.root / 'weights', self.root / 'in' / 'Archive')
        with patch.object(self.web, 'service', service), patch.object(self.web, 'UPLOAD_DIR', uploads), \
             patch.object(service, '_enhance_image', side_effect=self.enhance):
            response = self.web.app.test_client().post('/api/upscale', data={
                'source_mode': 'url', 'urls': urls, 'scale': '2', 'device_mode': 'cpu', **fields,
            })
        self.assertFalse(list(uploads.iterdir()), 'downloaded originals must always be cleaned up')
        return response

    def test_api_downloads_then_upscales_and_keeps_failures_and_duplicate_names(self):
        urls = [self.base_url + path for path in ('/photo.png', '/missing', '/redirect', '/html', '/named')]
        response = self.post_urls('\r\n\r\n'.join(urls), processing_mode='low', denoise_level='9')
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json
        self.assertEqual(payload['source_mode'], 'url')
        self.assertEqual(payload['denoise_level'], 2)
        self.assertEqual([r['input_name'] for r in payload['results']], ['photo.png', 'photo.png', '原图.png'])
        self.assertEqual([f['input_name'] for f in payload['failures']], [urls[1], urls[3]])
        self.assertEqual(len({r['output_name'] for r in payload['results']}), 3)
        for result in payload['results']:
            self.assertEqual((result['width'], result['height']), (18, 16))
            self.assertTrue(Path(result['output_path']).is_file())
            self.assertIsNone(result['archive_path'])

    def test_api_no_urls_or_too_many_urls_is_a_validation_error(self):
        for urls in ('\n \r\n', '\n'.join([self.base_url + '/photo.png'] * 101)):
            response = self.post_urls(urls)
            self.assertEqual(response.status_code, 400, response.text)

    def test_api_all_downloads_failed_includes_individual_details(self):
        with patch.object(RealESRGANBatchService, 'process_batch') as process:
            response = self.post_urls(f'{self.base_url}/missing\nfile:///tmp/a.png')
        self.assertEqual(response.status_code, 502)
        self.assertEqual(len(response.json['failures']), 2)
        self.assertEqual(response.json['results'], [])
        process.assert_not_called()

    def test_api_batch_byte_limit_still_processes_previously_downloaded_images(self):
        with patch.object(self.web, 'MAX_BATCH_BYTES', len(PNG)):
            response = self.post_urls(f'{self.base_url}/photo.png\n{self.base_url}/named')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(response.json['results']), 1)
        self.assertEqual(len(response.json['failures']), 1)

    def test_api_inference_error_also_cleans_up_downloads(self):
        with patch.object(RealESRGANBatchService, 'process_batch', side_effect=UpscaleRequestError('failed')):
            response = self.post_urls(self.base_url + '/photo.png')
        self.assertEqual(response.status_code, 500)


if __name__ == '__main__':
    unittest.main()
