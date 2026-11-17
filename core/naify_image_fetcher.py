from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from queue import Queue, Empty
from hashlib import sha256
from core.naify_cache import get_image_from_cache
from core.naify_api_work import generate_image
from core.naify_utility import print_log
from core import naify_cache
from core.naify_config import NaifyConfig, NaifySecrets
from fastapi import HTTPException
import threading
import os
import time


generation_request_queue = Queue()


def fetch_image(image_id: str, secrets: NaifySecrets, config: NaifyConfig, prompt: str, worker_thread: threading.Thread) -> bytes:
    # 캐시 확인과 파일 읽기는 하나의 잠금 안에서 처리한다.
    # 전체 삭제가 동시에 실행되어도 파일이 중간에 사라지지 않는다.
    image_bytes = get_image_from_cache(image_id)

    if image_bytes is not None:
        print_log(image_id, "Getting image from cache")
    else:
        if not worker_thread.is_alive():
            raise HTTPException(status_code=503, detail="Image generation worker stopped")
        print_log(image_id, "Queued for generation")
        future = Future()
        generation_request_queue.put((image_id, secrets, config, prompt, future))
        # 큐가 길어도 정상적인 생성 작업은 기다린다. 단, Worker가 죽으면 대기를 끝낸다.
        while True:
            try:
                image_bytes = future.result(timeout=1.0)
                break
            except FutureTimeoutError:
                if not worker_thread.is_alive():
                    raise HTTPException(status_code=503, detail="Image generation worker stopped")

    return image_bytes


def generation_request_queue_processor() -> None:
    """
    이미지 생성 요청 큐의 요청들을 관리하는 함수.
    캐시를 재조회해 중복 생성을 막는다.
    """

    print_log(content=f"Worker started | PID={os.getpid()} | Thread={threading.current_thread().name}")

    while True:
        image_id, secrets, config, prompt, future = generation_request_queue.get()

        try:
            # 큐에서 꺼낸 시점에도 캐시를 재조회해 중복 생성을 방지한다.
            image_bytes = get_image_from_cache(image_id)

            if image_bytes is not None:
                print_log(image_id=image_id, content="Found in cache. Serving existing image from cache.")

            else:
                print_log(image_id=image_id, content="Generating...")

                # 시작 시각 기록
                process_initiation_time = time.perf_counter()

                image_bytes = generate_image(image_id=image_id, naify_secrets=secrets, naify_config=config, prompt=prompt)

                # 소요 시간 계산
                process_termination_time = time.perf_counter()
                total_elapsed = process_termination_time - process_initiation_time

                pending_count = generation_request_queue.qsize()
                print_log(image_id=image_id, content=f"Generated: {round(total_elapsed, 2)}sec | {pending_count} more left")

                naify_cache.save_image_to_cache(image_id, image_bytes)
                naify_cache.trim_cache(config.cache)

            future.set_result(image_bytes)

        except Exception as e:
            future.set_exception(e)

        finally:
            generation_request_queue.task_done()

def cancel_pending_requests() -> int:
    """
    큐에서 아직 Worker가 가져가지 않은 요청을 취소한다.
    현재 생성 중인 작업은 큐에서 이미 빠져 있으므로 영향을 받지 않는다.
    """
    cancelled_count = 0

    while True:
        try:
            image_id, secrets, config, prompt, future = generation_request_queue.get_nowait()
        except Empty:
            break

        try:
            future.set_exception(
                HTTPException(status_code=503, detail="Image generation request cancelled")
            )
            cancelled_count += 1
        finally:
            # get()으로 꺼낸 모든 작업은 반드시 완료 처리해야 종료 대기가 풀린다.
            generation_request_queue.task_done()

    print_log(content=f"Cancelled {cancelled_count} pending generation requests")

    return cancelled_count


def make_image_id(prompt: str) -> str:
    image_id = sha256(prompt.encode("utf-8")).hexdigest()

    return image_id
