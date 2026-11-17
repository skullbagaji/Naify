import os
import sys
from pathlib import Path
from typing import Final
from core.naify_utility import print_log

def get_data_dir() -> Path:

    data_dir: Path

    if os.name == "nt": # Windows 환경
        data_dir = Path(os.environ["LOCALAPPDATA"]) / "Naify"
    else:
        data_dir = Path.home() / ".local" / "share" / "naify"

    return data_dir


def get_resource_dir() -> Path:

    resource_dir: Path

    if getattr(sys, "frozen", False): # PyInstaller 실행 환경
        resource_dir = Path(getattr(sys, "_MEIPASS"))
    else: # 일반 Python 실행 환경
        resource_dir = Path(__file__).resolve().parent.parent

    return resource_dir


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
        

DATA_DIR: Final[Path] = get_data_dir()

CONFIG_PATH: Final[Path] = DATA_DIR / "config.json"

SECRETS_PATH: Final[Path] = DATA_DIR / "secrets.json"

CACHE_DIR: Final[Path] = DATA_DIR / "cache"

RESOURCE_DIR: Final[Path] = get_resource_dir()

WEB_DIR: Final[Path] = RESOURCE_DIR / "web"