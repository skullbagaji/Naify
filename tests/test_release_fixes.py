import asyncio
import io
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException

from core import naify_api_work, naify_config, naify_image_fetcher, naify_utility
import main


def image_archive(filename: str = "image.webp", data: bytes = b"RIFF\x04\x00\x00\x00WEBP") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(filename, data)
    return buffer.getvalue()


class ReleaseFixTests(unittest.TestCase):
    def test_resolution_keeps_last_edited_height(self):
        generation = naify_config.GenerationConfig(width=3008, height=1000)
        generation.fit_resolution("height")
        self.assertEqual((generation.width, generation.height), (2496, 1024))

    def test_resolution_keeps_last_edited_width(self):
        generation = naify_config.GenerationConfig(width=3008, height=1536)
        generation.fit_resolution("width")
        self.assertEqual((generation.width, generation.height), (3008, 512))

    def test_resolution_normalizes_to_multiples_of_64(self):
        generation = naify_config.GenerationConfig(width=1000, height=999999)
        generation.fit_resolution("width")
        self.assertEqual((generation.width, generation.height), (1024, 2496))

    def test_api_key_validation_distinguishes_auth_and_outage(self):
        with patch.object(naify_api_work.httpx, "get", return_value=Mock(status_code=200)):
            self.assertEqual(naify_api_work.validate_api_key("key"), "valid")
        with patch.object(naify_api_work.httpx, "get", return_value=Mock(status_code=401)):
            self.assertEqual(naify_api_work.validate_api_key("key"), "invalid")
        with patch.object(naify_api_work.httpx, "get", return_value=Mock(status_code=500)):
            self.assertEqual(naify_api_work.validate_api_key("key"), "unavailable")
        self.assertEqual(naify_api_work.validate_api_key(""), "missing")

    def test_invalid_api_key_does_not_replace_existing_key(self):
        request = SimpleNamespace(headers={"origin": "http://127.0.0.1:8000"})
        original = main.naify_secrets
        try:
            main.naify_secrets = naify_config.NaifySecrets(nai_api_key="previous")
            with patch.object(main.naify_api_work, "validate_api_key", return_value="invalid"), \
                    patch.object(main, "save_secrets") as save:
                with self.assertRaises(HTTPException) as raised:
                    main.update_secrets(naify_config.NaifySecrets(nai_api_key="wrong"), request)
            self.assertEqual(raised.exception.status_code, 401)
            self.assertEqual(main.naify_secrets.nai_api_key, "previous")
            save.assert_not_called()
        finally:
            main.naify_secrets = original

    def test_config_endpoint_prioritizes_last_edited_dimension(self):
        request = SimpleNamespace(headers={"x-naify-resolution-priority": "height"})
        original = main.naify_config
        try:
            config = naify_config.NaifyConfig(generation={"width": 3008, "height": 1000})
            with patch.object(main, "save_config") as save, \
                    patch.object(main.naify_cache, "trim_cache") as trim:
                main.update_config(config, request)
            self.assertEqual((config.generation.width, config.generation.height), (2496, 1024))
            save.assert_called_once_with(config)
            trim.assert_called_once_with(config.cache)
        finally:
            main.naify_config = original

    def test_old_config_with_seed_still_loads(self):
        config = naify_config.NaifyConfig.model_validate({"generation": {"seed": 12345}})
        self.assertNotIn("seed", config.generation.model_dump())

    def test_log_waiter_wakes_up_on_new_log(self):
        seen = naify_utility.last_log_id
        result = []
        waiter = threading.Thread(target=lambda: result.extend(naify_utility.wait_for_logs(seen)))
        waiter.start()
        time.sleep(0.05)
        naify_utility.print_log(content="test log notification")
        waiter.join(timeout=0.5)
        self.assertFalse(waiter.is_alive(), "log waiter should wake before its one-second timeout")
        self.assertTrue(result)

    def test_non_json_api_error_is_translated_to_502(self):
        response = Mock(status_code=503, text="Service Unavailable")
        response.json.side_effect = ValueError("not JSON")
        with patch.object(naify_api_work.httpx, "post", return_value=response):
            with self.assertRaises(HTTPException) as raised:
                naify_api_work.generate_image("id", naify_config.NaifySecrets(), naify_config.NaifyConfig(), "prompt")
        self.assertEqual(raised.exception.status_code, 502)

    def test_api_auth_failure_is_502(self):
        response = Mock(status_code=401, text="Unauthorized")
        response.json.return_value = {"message": "Unauthorized"}
        with patch.object(naify_api_work.httpx, "post", return_value=response):
            with self.assertRaises(HTTPException) as raised:
                naify_api_work.generate_image("id", naify_config.NaifySecrets(), naify_config.NaifyConfig(), "prompt")
        self.assertEqual((raised.exception.status_code, raised.exception.detail), (502, "NovelAI authentication failed"))

    def test_api_generation_limit_is_503_without_retry(self):
        response = Mock(status_code=429, text="Concurrent generation locked")
        response.json.return_value = {"message": "Concurrent generation locked"}
        with patch.object(naify_api_work.httpx, "post", return_value=response) as post:
            with self.assertRaises(HTTPException) as raised:
                naify_api_work.generate_image("id", naify_config.NaifySecrets(), naify_config.NaifyConfig(), "prompt")
        self.assertEqual(raised.exception.status_code, 503)
        post.assert_called_once()

    def test_api_timeout_and_connection_errors(self):
        cases = [(naify_api_work.httpx.ReadTimeout("timeout"), 504),
                 (naify_api_work.httpx.ConnectError("connection failed"), 502)]
        for error, status_code in cases:
            with self.subTest(error=type(error).__name__):
                with patch.object(naify_api_work.httpx, "post", side_effect=error) as post:
                    with self.assertRaises(HTTPException) as raised:
                        naify_api_work.generate_image("id", naify_config.NaifySecrets(), naify_config.NaifyConfig(), "prompt")
                self.assertEqual(raised.exception.status_code, status_code)
                post.assert_called_once()

    def test_invalid_image_archives_return_502(self):
        invalid_archives = [
            b"not a zip",
            image_archive("notes.txt", b"some notes"),
            image_archive(data=b"fake webp"),
        ]
        for archive in invalid_archives:
            with self.subTest(archive=archive[:12]):
                response = Mock(status_code=200, content=archive)
                with patch.object(naify_api_work.httpx, "post", return_value=response):
                    with self.assertRaises(HTTPException) as raised:
                        naify_api_work.generate_image("id", naify_config.NaifySecrets(), naify_config.NaifyConfig(), "prompt")
                self.assertEqual(raised.exception.status_code, 502)

    def test_zip_extraction_uses_webp_instead_of_first_file(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("meta.txt", "metadata")
            archive.writestr("image.webp", b"RIFF\x04\x00\x00\x00WEBP")
        self.assertEqual(naify_utility.extract_image_from_zip(buffer.getvalue()), b"RIFF\x04\x00\x00\x00WEBP")

    def test_successful_201_image_response(self):
        response = Mock(status_code=201, content=image_archive())
        with patch.object(naify_api_work.httpx, "post", return_value=response):
            result = naify_api_work.generate_image("id", naify_config.NaifySecrets(), naify_config.NaifyConfig(), "prompt")
        self.assertTrue(result.startswith(b"RIFF"))

    def test_generation_request_omits_seed(self):
        response = Mock(status_code=200, content=b"zip-data")
        with patch.object(naify_api_work.httpx, "post", return_value=response) as post, \
                patch.object(naify_api_work, "extract_image_from_zip", return_value=b"webp"):
            naify_api_work.generate_image("id", naify_config.NaifySecrets(), naify_config.NaifyConfig(), "prompt")
        self.assertNotIn("seed", post.call_args.kwargs["json"]["parameters"])

    def test_atomic_save_keeps_previous_file_on_replace_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "config.json"
            destination.write_text("original", encoding="utf-8")
            with patch.object(naify_config.os, "replace", side_effect=OSError("replace failed")):
                with self.assertRaises(OSError):
                    naify_config.save_json_atomically(destination, "replacement")
            self.assertEqual(destination.read_text(encoding="utf-8"), "original")
            self.assertEqual(list(Path(directory).iterdir()), [destination])
            naify_config.save_json_atomically(destination, "replacement")
            self.assertEqual(destination.read_text(encoding="utf-8"), "replacement")

    def test_dead_worker_returns_503(self):
        stopped_worker = threading.Thread(target=lambda: None)
        with patch.object(naify_image_fetcher, "get_image_from_cache", return_value=None):
            with self.assertRaises(HTTPException) as raised:
                naify_image_fetcher.fetch_image(
                    "id", naify_config.NaifySecrets(), naify_config.NaifyConfig(),
                    "prompt", stopped_worker
                )
        self.assertEqual(raised.exception.status_code, 503)


class ShutdownStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_checks_stored_api_key_once(self):
        original = main.naify_secrets
        try:
            main.naify_secrets = naify_config.NaifySecrets(nai_api_key="stored-key")
            with patch.object(main.naify_api_work, "validate_api_key", return_value="valid") as validate, \
                    patch.object(main.threading, "Timer"):
                async with main.lifespan(main.app):
                    self.assertEqual(main.app.state.api_key_status, "valid")
            validate.assert_called_once_with("stored-key")
        finally:
            main.naify_secrets = original

    async def test_log_stream_closes_on_shutdown(self):
        request = SimpleNamespace(
            app=main.app,
            is_disconnected=AsyncMock(return_value=False)
        )
        release_waiter = threading.Event()

        def wait_for_logs_until_released(last_seen_id):
            release_waiter.wait(timeout=2.0)
            return [(1, "test log")]

        main.app.state.shutting_down = False
        try:
            with patch.object(main, "wait_for_logs", side_effect=wait_for_logs_until_released):
                response = await main.stream_logs(request)
                next_event = asyncio.create_task(response.body_iterator.__anext__())
                await asyncio.sleep(0.05)
                main.app.state.shutting_down = True
                release_waiter.set()

                with self.assertRaises(StopAsyncIteration):
                    await asyncio.wait_for(next_event, timeout=2.0)
        finally:
            release_waiter.set()
            main.app.state.shutting_down = False


if __name__ == "__main__":
    unittest.main()
