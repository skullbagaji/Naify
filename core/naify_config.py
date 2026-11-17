# 상수 및 설정 
from pydantic import BaseModel, field_validator
from pydantic import ConfigDict
from pydantic import Field
import os
import tempfile
from pathlib import Path
from core.naify_paths import CONFIG_PATH, SECRETS_PATH

class NaifySecrets(BaseModel):
    nai_api_key: str = ""


class GenerationPrompt(BaseModel):

    model_config = ConfigDict(
        validate_default=True,
        validate_assignment=True
    )

    base_prompt: str = ""
    negative_prompt: str = ""


class GenerationConfig(BaseModel):

    model_config = ConfigDict(
        validate_default=True,
        validate_assignment=True
    )

    model: str = "nai-diffusion-4-5-full"
    width: int = 1216
    height: int = 832
    steps: int = 28
    prompt_guidance: float = 7
    sampler: str = "k_euler_ancestral"
    prompt_guidance_rescale: float = 0.4
    noise_schedule: str = "karras"

    @field_validator("width", "height")
    @classmethod
    def normalize_resolution(cls, value: int) -> int:
        normalized_value: int = min(3008, max(512, ((value + 32) // 64) * 64))
        
        return normalized_value

    def fit_resolution(self, preferred: str = "height") -> None:
        """마지막으로 변경한 변을 유지하고 다른 변을 최대 합계에 맞춘다."""
        if self.width + self.height <= 3520:
            return
        if preferred == "width":
            self.height = 3520 - self.width
        else:
            self.width = 3520 - self.height

    @field_validator("steps")
    @classmethod
    def normalize_step_count(cls, value: int):
        if value > 50 : value = 50
        if value < 1 : value = 1

        return value

    @field_validator("prompt_guidance")
    @classmethod
    def normalize_prompt_guidance(cls, value: float):
        if value > 10 : value = 10
        if value < 0 : value = 0

        return value

    @field_validator("prompt_guidance_rescale")
    @classmethod
    def normalize_prompt_guidance_rescale(cls, value: float):
        if value > 1 : value = 1
        if value < 0 : value = 0

        return value


class CacheConfig(BaseModel):

    model_config = ConfigDict(
        validate_default=True,
        validate_assignment=True
    )

    max_cache_size: int = Field(
        default=100 * 1024 * 1024,
        gt=0
    )

    max_images: int = Field(
        default=100,
        gt=0
    )


class NaifyConfig(BaseModel):
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    prompt: GenerationPrompt = Field(default_factory=GenerationPrompt)
    cache: CacheConfig = Field(default_factory=CacheConfig)


def save_json_atomically(path: Path, content: str) -> None:
    """기존 파일을 유지한 채 새 내용을 기록한 후 한 번에 교체한다."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def save_config(naify_config: NaifyConfig) -> None:
    save_json_atomically(CONFIG_PATH, naify_config.model_dump_json(indent=4))


def load_config() -> NaifyConfig:
    naify_config: NaifyConfig

    if not CONFIG_PATH.exists():
        naify_config = NaifyConfig()
        save_config(naify_config)
        return naify_config

    with open(CONFIG_PATH, "r", encoding="utf-8") as json_file:
        config_json: str = json_file.read()

    naify_config = NaifyConfig.model_validate_json(config_json)
    naify_config.generation.fit_resolution()

    return naify_config


def save_secrets(naify_secrets: NaifySecrets) -> None:
    save_json_atomically(SECRETS_PATH, naify_secrets.model_dump_json(indent=4))


def load_secrets() -> NaifySecrets:
    naify_secrets: NaifySecrets

    if not SECRETS_PATH.exists():
        naify_secrets = NaifySecrets()
        save_secrets(naify_secrets)
        return naify_secrets

    with open(SECRETS_PATH, "r", encoding="utf-8") as json_file:
        secrets_json: str = json_file.read()

    naify_secrets = NaifySecrets.model_validate_json(secrets_json)

    return naify_secrets
