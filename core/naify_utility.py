import io
from datetime import datetime
import zipfile
import uvicorn
import webbrowser
import fastapi
from collections import deque
import threading
import sys


log_history = deque(maxlen=300)
log_condition = threading.Condition()
last_log_id = 0


def print_log(image_id:str = "SERVER", content: str = "") -> None:
    log: str
    now = datetime.now()
    timestamp = now.strftime("%H:%M:%S.%f")[:-3]

    global last_log_id

    log = f"[{timestamp}] {image_id[:6]} : {content}"

    with log_condition:
        last_log_id += 1

        log_history.append((last_log_id, log))

        log_condition.notify_all()

    if sys.stdout is not None:
        print(log)


def wait_for_logs(last_seen_id: int) -> list[tuple[int, str]]:
    with log_condition:

        log_condition.wait_for(
            lambda: last_log_id > last_seen_id,
            timeout=1.0
        )

        new_logs = [
            (log_id, log)
            for log_id, log in log_history
            if log_id > last_seen_id
        ]

        return new_logs



def extract_image_from_zip(zip_data: bytes) -> bytes:
    """
    ZIP 바이너리 데이터에서 WebP 이미지 파일을 추출한다.

    Args:
        zip_data: ZIP 파일의 바이너리 데이터.

    Returns:
        bytes: 압축 해제된 이미지 데이터.
    """

    zip_file = io.BytesIO(zip_data)
    with zipfile.ZipFile(zip_file, "r") as response_zip:
        image_files = [
            entry for entry in response_zip.infolist()
            if not entry.is_dir() and entry.filename.lower().endswith(".webp")
        ]
        if not image_files:
            raise ValueError("No WebP image found in NovelAI response")
        image_bytes = response_zip.read(image_files[0])
        if len(image_bytes) < 12 or image_bytes[:4] != b"RIFF" or image_bytes[8:12] != b"WEBP":
            raise ValueError("Invalid WebP image data")
    return image_bytes


def open_manager() -> None:
    webbrowser.open("http://127.0.0.1:8000/")


def shutdown_server(app: fastapi.FastAPI, server: uvicorn.Server) -> None:
    from core import naify_image_fetcher

    # 현재 처리 중인 이미지 요청이 모두 끝날 때까지 대기
    with app.state.request_condition:

        app.state.request_condition.wait_for(
            lambda: app.state.active_requests == 0
        )

    # Worker가 비정상 종료된 경우에는 큐가 영원히 비워지지 않을 수 있다.
    queue = naify_image_fetcher.generation_request_queue
    worker = app.state.generation_worker
    with queue.all_tasks_done:
        while queue.unfinished_tasks and worker.is_alive():
            queue.all_tasks_done.wait(timeout=1.0)

        worker_failed = bool(queue.unfinished_tasks)
        if worker_failed:
            print_log(content="Generation worker stopped before queue was drained")

    if not worker_failed:
        print_log(content="All generation tasks completed")

    server.should_exit = True

    return
