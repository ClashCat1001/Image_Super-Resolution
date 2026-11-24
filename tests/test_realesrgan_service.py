import importlib.util
import io
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from app.realesrgan_service import (
    MODEL_NAME, BatchImage, RealESRGANBatchService, UpscaleRequestError,
)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.service = RealESRGANBatchService(
            self.root / 'out', self.root / 'weights', self.root / 'in' / 'Archive',
        )

    def tearDown(self):
        self.temp.cleanup()

    def image(self, name='a.png', value=100, archive=False):
        source = self.root / 'in' / name
        source.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(source), np.full((8, 9, 3), value, np.uint8))
        return BatchImage(source, name, archive_subdir=Path() if archive else None)

    @staticmethod
    def enhance(image, scale, denoise_strength, requested_device_type, mps_disabled):
        out = cv2.resize(image, None, fx=scale, fy=scale)
        return out, 'cpu', None, mps_disabled

    def test_batch_prefetch_failures_and_archiving(self):
        good = self.image(archive=True)
        bad = BatchImage(self.root / 'missing.png', 'missing.png')
        other = self.image('b.png')
        with patch.object(self.service, '_enhance_image', side_effect=self.enhance):
            results, failures = self.service.process_batch([good, bad, other], 3, 7, 'cpu', 'low')
        self.assertEqual([r.input_name for r in results], ['a.png', 'b.png'])
        self.assertEqual([f['input_name'] for f in failures], ['missing.png'])
        self.assertFalse(good.source_path.exists())
        self.assertTrue(Path(results[0].archive_path).exists())
        self.assertEqual((results[0].width, results[0].height), (27, 24))
        self.assertEqual(results[0].denoise_level, 2)
        self.assertIn('-d2x-low', results[0].output_name)
        self.assertEqual(results[0].to_dict('/preview')['model_name'], MODEL_NAME)

    def test_write_failure_does_not_archive_or_stop_batch(self):
        a, b = self.image(archive=True), self.image('b.png', archive=True)
        real_write = cv2.imwrite
        def write(path, image):
            return False if Path(path).stem.startswith('a-') else real_write(path, image)
        with patch.object(self.service, '_enhance_image', side_effect=self.enhance), patch('cv2.imwrite', side_effect=write):
            results, failures = self.service.process_batch([a, b], 2, 5, 'cpu')
        self.assertEqual(len(results), 1)
        self.assertEqual(failures[0]['input_name'], 'a.png')
        self.assertTrue(a.source_path.exists())
        self.assertFalse(b.source_path.exists())

    def test_blend_precision_alpha_and_sizes(self):
        for dtype in (np.uint8, np.uint16):
            for channels in (3, 4):
                for scale in (2, 3, 4):
                    image = np.full((5, 7, channels), 40, dtype)
                    enhanced = np.full((5 * scale, 7 * scale, channels), 120, dtype)
                    output = self.service._blend_low_processing(image, enhanced)
                    self.assertEqual(output.dtype, dtype)
                    np.testing.assert_array_equal(output[..., :3], 60)
                    if channels == 4:
                        np.testing.assert_array_equal(output[..., 3], 40)

    def test_standard_keeps_denoise_and_naming_and_collision_suffix(self):
        item = self.image()
        with patch.object(self.service, '_enhance_image', side_effect=self.enhance) as enhance:
            first, _ = self.service.process_batch([item], 4, 8, 'cpu')
            second, _ = self.service.process_batch([item], 4, 8, 'cpu')
        self.assertEqual(enhance.call_args.kwargs['denoise_strength'], .8)
        self.assertEqual(first[0].output_name, 'a-4x-d8x.png')
        self.assertEqual(second[0].output_name, 'a-4x-d8x-1.png')

    def test_device_priority_and_explicit_selection(self):
        for cuda in (True, False):
            for metal in (True, False):
                with patch('torch.cuda.is_available', return_value=cuda), patch.object(self.service, '_mps_available', return_value=metal):
                    self.assertEqual(self.service._resolve_device_type('auto')[0], 'cuda' if cuda else 'mps' if metal else 'cpu')
                    self.assertEqual(self.service._resolve_device_type('cpu'), ('cpu', None))
                    self.assertEqual(self.service._resolve_device_type('metal')[0], 'mps' if metal else 'cpu')
                    self.assertEqual(self.service._resolve_device_type('cuda')[0], 'cuda' if cuda else 'cpu')

    def test_metal_black_output_falls_back_and_disables_for_next_image(self):
        class Upsampler:
            def __init__(self, device):
                self.device = device
            def enhance(self, image, outscale):
                assert torch.is_inference_mode_enabled()
                self.img = torch.ones(1)
                self.output = torch.ones(1)
                out = cv2.resize(image, None, fx=outscale, fy=outscale)
                return (np.zeros_like(out) if self.device == 'mps' else out), None
        devices = []
        helpers = []
        def get(strength, device):
            devices.append(device)
            helper = Upsampler(device)
            helpers.append(helper)
            return helper
        with patch.object(self.service, '_get_upsampler', side_effect=get), patch.object(self.service, '_resolve_device_type', return_value=('mps', None)):
            results, failures = self.service.process_batch([self.image(), self.image('b.png')], 2, 5)
        self.assertEqual(devices, ['mps', 'cpu', 'cpu'])
        self.assertEqual(len(results), 2)
        self.assertFalse(failures)
        self.assertEqual(self.service.last_runtime_profile, 'metal-fallback')
        self.assertEqual(len(self.service.last_warnings), 1)
        self.assertTrue(all(h.img is None and h.output is None for h in helpers))

    def test_tiled_geometry_and_pixel_assembly(self):
        class ResizeUpsampler:
            def enhance(self, image, outscale):
                return cv2.resize(image, None, fx=outscale, fy=outscale, interpolation=cv2.INTER_NEAREST), None
        image = np.random.default_rng(4).integers(40, 255, (13, 17, 4), dtype=np.uint8)
        for scale in (2, 3, 4):
            output = self.service._enhance_image_tiled(image, ResizeUpsampler(), scale, 6, 3)
            expected = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            np.testing.assert_array_equal(output, expected)

    def test_inference_and_metadata_are_serialized_between_batches(self):
        active = 0
        max_active = 0
        def enhance(**kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(active, max_active)
            time.sleep(.03)
            output = self.enhance(**kwargs)
            active -= 1
            return output
        item = self.image()
        with patch.object(self.service, '_enhance_image', side_effect=enhance):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(self.service.process_batch, [item], 2, 5, 'cpu') for _ in range(2)]
                batches = [f.result() for f in futures]
        self.assertEqual(max_active, 1)
        self.assertNotEqual(batches[0][0][0].output_path, batches[1][0][0].output_path)

    def test_read_and_write_overlap_inference(self):
        write_started, next_read = threading.Event(), threading.Event()
        real_read, real_save = self.service._read_image, self.service._save_result
        def read(item):
            if item.input_name == 'b.png':
                next_read.set()
            return real_read(item)
        def save(*args):
            write_started.set()
            return real_save(*args)
        calls = 0
        def enhance(**kwargs):
            nonlocal calls
            calls += 1
            self.assertTrue(next_read.wait(2))
            if calls == 2:
                self.assertTrue(write_started.wait(2))
            return self.enhance(**kwargs)
        items = [self.image(), self.image('b.png')]
        with patch.object(self.service, '_read_image', side_effect=read), patch.object(self.service, '_save_result', side_effect=save), patch.object(self.service, '_enhance_image', side_effect=enhance):
            results, failures = self.service.process_batch(items, 2, 5, 'cpu')
        self.assertEqual(len(results), 2)
        self.assertFalse(failures)

    def test_cuda_cache_is_flushed_once_per_batch(self):
        def enhance(**kwargs):
            output, _, warning, disabled = self.enhance(**kwargs)
            return output, 'cuda', warning, disabled
        with patch.object(self.service, '_resolve_device_type', return_value=('cuda', None)), patch.object(self.service, '_enhance_image', side_effect=enhance), patch('torch.cuda.empty_cache') as flush:
            results, failures = self.service.process_batch([self.image(), self.image('b.png')], 2, 5, 'cuda')
        self.assertEqual(len(results), 2)
        self.assertFalse(failures)
        flush.assert_called_once()
        self.assertEqual(self.service.last_device.type, 'cuda')

    def test_empty_and_invalid_requests(self):
        self.assertEqual(self.service.process_batch([], 2, 5), ([], []))
        with self.assertRaises(UpscaleRequestError):
            self.service.process_batch([], 2, 5, processing_mode='unknown')


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('webapp', Path(__file__).resolve().parents[1] / 'app.py')
        cls.web = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.web)

    def test_api_validation_default_and_low_metadata(self):
        web = self.web
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = RealESRGANBatchService(root / 'out', root / 'weights', root / 'in' / 'Archive')
            with patch.object(web, 'service', service), patch.object(web, 'UPLOAD_DIR', root), patch.object(service, '_enhance_image', side_effect=ServiceTests.enhance):
                client = web.app.test_client()
                self.assertEqual(client.post('/api/upscale', data={'processing_mode': 'bad'}).status_code, 400)
                self.assertEqual(client.get('/api/runtime').json['model_name'], MODEL_NAME)
                self.assertIn('processing-mode', client.get('/').text)
                for mode in (None, 'low'):
                    image = np.full((8, 9, 3), 100, np.uint8)
                    data = {'device_mode': 'cpu', 'scale': '2', 'denoise_level': '9', 'files': (io.BytesIO(cv2.imencode('.png', image)[1].tobytes()), 'test.png')}
                    if mode:
                        data['processing_mode'] = mode
                    response = client.post('/api/upscale', data=data)
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json['denoise_level'], 2 if mode else 9)
                    self.assertEqual(response.json['requested_denoise_level'], 9)
                    self.assertEqual(response.json['results'][0]['processing_mode'], mode or 'standard')
                self.assertFalse(list(root.glob('*.png')))

    def test_concurrent_folder_requests_do_not_process_same_original_twice(self):
        web = self.web
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = RealESRGANBatchService(root / 'out', root / 'weights', root / 'in' / 'Archive')
            cv2.imwrite(str(root / 'in' / 'one.png'), np.full((8, 9, 3), 100, np.uint8))
            def post():
                with web.app.test_client() as client:
                    return client.post('/api/upscale', data={'source_mode': 'input-folder', 'device_mode': 'cpu'})
            with patch.object(web, 'service', service), patch.object(web, 'INPUT_DIR', root / 'in'), patch.object(service, '_enhance_image', side_effect=ServiceTests.enhance):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    responses = list(pool.map(lambda _: post(), range(2)))
            self.assertEqual(sorted(r.status_code for r in responses), [200, 400])
            self.assertEqual(len(list((root / 'out').glob('*.png'))), 1)
            self.assertTrue((root / 'in' / 'Archive' / 'one.png').exists())


if __name__ == '__main__':
    unittest.main()
