import unittest
from concurrent.futures import Future
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from core import naify_image_fetcher
import main


class PendingGenerationCancellationTests(unittest.TestCase):
    def setUp(self):
        self.original_queue = naify_image_fetcher.generation_request_queue
        self.queue = Queue()
        naify_image_fetcher.generation_request_queue = self.queue

    def tearDown(self):
        naify_image_fetcher.generation_request_queue = self.original_queue

    def enqueue(self, image_id):
        future = Future()
        self.queue.put((image_id, None, None, "prompt", future))
        return future

    def test_pending_requests_are_cancelled_and_queue_accounting_finishes(self):
        first = self.enqueue("first")
        second = self.enqueue("second")

        cancelled_count = naify_image_fetcher.cancel_pending_requests()

        self.assertEqual(cancelled_count, 2)
        self.assertEqual(self.queue.qsize(), 0)
        self.assertEqual(self.queue.unfinished_tasks, 0)
        for future in (first, second):
            with self.assertRaises(HTTPException) as raised:
                future.result()
            self.assertEqual(raised.exception.status_code, 503)
            self.assertEqual(raised.exception.detail, "Image generation request cancelled")

    def test_in_progress_request_is_not_cancelled(self):
        active = self.enqueue("active")
        self.queue.get_nowait()  # Worker가 이미 가져간 요청을 모사한다.
        pending = self.enqueue("pending")

        cancelled_count = naify_image_fetcher.cancel_pending_requests()

        self.assertEqual(cancelled_count, 1)
        self.assertFalse(active.done())
        with self.assertRaises(HTTPException):
            pending.result()
        active.set_result(b"webp")
        self.queue.task_done()
        self.assertEqual(self.queue.unfinished_tasks, 0)

    def test_empty_queue_is_safe(self):
        self.assertEqual(naify_image_fetcher.cancel_pending_requests(), 0)
        self.assertEqual(self.queue.unfinished_tasks, 0)

    def test_endpoint_rejects_untrusted_requests(self):
        request = SimpleNamespace(headers={"origin": "http://evil.example", "x-naify-action": "cancel-pending"})
        with self.assertRaises(HTTPException) as raised:
            main.cancel_pending_generation_requests(request)
        self.assertEqual(raised.exception.status_code, 403)

    def test_endpoint_returns_cancelled_count(self):
        self.enqueue("one")
        request = SimpleNamespace(headers={"origin": "http://127.0.0.1:8000", "x-naify-action": "cancel-pending"})
        original_shutting_down = main.app.state.shutting_down
        try:
            main.app.state.shutting_down = False
            result = main.cancel_pending_generation_requests(request)
        finally:
            main.app.state.shutting_down = original_shutting_down
        self.assertEqual(result, {"cancelled_count": 1})


if __name__ == "__main__":
    unittest.main()
