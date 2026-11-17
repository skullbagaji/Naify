from threading import RLock
from core.naify_utility import print_log
from core.naify_config import CacheConfig
from core.naify_paths import CACHE_DIR


# 생성 Worker와 웹 관리자에서 동시에 캐시 파일을 건드리지 않도록 보호.
cache_lock = RLock()


def get_image_from_cache(image_id: str) -> bytes | None:
    with cache_lock:
        image_path = CACHE_DIR / f"{image_id}.webp"
        if not image_path.is_file():
            return None
        return image_path.read_bytes()


def save_image_to_cache(image_id: str, image_bytes: bytes) -> None:
    with cache_lock:
        image_path = CACHE_DIR / f"{image_id}.webp"
        image_path.write_bytes(image_bytes)


def get_cache_size() -> int:
    with cache_lock:
        cache_size = sum(image.stat().st_size for image in CACHE_DIR.glob("*.webp"))
        return cache_size


def get_cache_status() -> dict:
    with cache_lock:
        images = []
        for image in CACHE_DIR.glob("*.webp"):
            image_stat = image.stat()
            images.append({
                "id": image.stem,
                "size": image_stat.st_size,
                "modified_at": image_stat.st_mtime
            })

        images.sort(key=lambda image: image["modified_at"], reverse=True)

        cache_status = {
            "image_count": len(images),
            "cache_size": sum(image["size"] for image in images),
            "images": images
        }

        return cache_status


def clear_cache() -> int:
    with cache_lock:
        deleted_count = 0
        for image in CACHE_DIR.glob("*.webp"):
            image.unlink()
            deleted_count += 1

        print_log(content=f"Cleared cache: {deleted_count} images")
        return deleted_count


def trim_cache(cache_config: CacheConfig) -> None:
    with cache_lock:
        cache_images = sorted(
            CACHE_DIR.glob("*.webp"),
            key=lambda image: image.stat().st_mtime,
            reverse=True
        )
        cache_size = sum(image.stat().st_size for image in cache_images)

        while (cache_size > cache_config.max_cache_size or len(cache_images) > cache_config.max_images) and cache_images:
            oldest_image = cache_images.pop()
            cache_size -= oldest_image.stat().st_size
            oldest_image.unlink()

            print_log(content=f"Deleted cache: {oldest_image.name}")
