import fastapi
import httpx
import zipfile
from typing import Final

from core.naify_utility import extract_image_from_zip, print_log
from core.naify_config import NaifyConfig, NaifySecrets


NAI_API_URL: Final[str] = "https://image.novelai.net/ai/generate-image"


def validate_api_key(api_key: str) -> str:
    """인증 확인: valid / invalid / unavailable / missing."""
    if not api_key.strip():
        return "missing"
    try:
        response = httpx.get(
            "https://image.novelai.net/user/subscription",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0
        )
    except httpx.RequestError:
        return "unavailable"

    if response.status_code == 200:
        return "valid"
    if response.status_code in (401, 403):
        return "invalid"
    return "unavailable"


def get_anlas_balance(naify_secrets: NaifySecrets) -> int | None:
    """
    NovelAI API에서 현재 남은 Anlas 잔액을 조회한다.

    Returns:
        int: 구독 Anlas와 구매 Anlas의 합계.
        None: 조회 실패 시 반환.
    """
    try:
        subscription_info_response = httpx.get(
            "https://image.novelai.net/user/subscription",
            headers={"Authorization": f"Bearer {naify_secrets.nai_api_key}"},
            timeout=10.0
        )
        subscription_info_response.raise_for_status()
        anlas_data = subscription_info_response.json().get("trainingStepsLeft")
        remaining_anlas = anlas_data["fixedTrainingStepsLeft"] + anlas_data["purchasedTrainingSteps"]
    except Exception as e:
        print_log(content=f"잔여 Anlas 조회 실패: {e}")
        return None
    else:
        return remaining_anlas


def generate_image(image_id: str, naify_secrets: NaifySecrets, naify_config: NaifyConfig, prompt: str) -> bytes:

    prompt = prompt.replace("_", " ")
    prompt = prompt.replace(",", ", ")
    prompt = prompt.replace(",  ", ", ")

    try:
        generation_response = httpx.post(
            timeout=120.0,
            url=NAI_API_URL,
            headers={"Authorization": f"Bearer {naify_secrets.nai_api_key}"},
            json = {
                "action": "generate",
                "input": naify_config.prompt.base_prompt,
                "model": naify_config.generation.model,
                "parameters": {
                    "width": naify_config.generation.width,
                    "height": naify_config.generation.height,
                    "n_samples": 1,
                    "negative_prompt": naify_config.prompt.negative_prompt,
                    "steps": naify_config.generation.steps,
                    "scale": naify_config.generation.prompt_guidance,
                    "sampler": naify_config.generation.sampler,
                    "noise_schedule": naify_config.generation.noise_schedule,
                    "cfg_rescale": naify_config.generation.prompt_guidance_rescale,
                    "params_version": 3,
                    "image_format": "webp",
                    "v4_prompt": {
                        "caption": {
                            "base_caption": naify_config.prompt.base_prompt,
                            "char_captions": [
                                {
                                    "char_caption": prompt,
                                    "centers": [
                                        {
                                            "x": 0.5,
                                            "y": 0.5
                                        }
                                    ]
                                }
                            ]
                        },
                        "use_coords": False,
                        "use_order": True
                    },
                    "v4_negative_prompt": {
                        "caption": {
                            "base_caption": naify_config.prompt.negative_prompt,
                            "char_captions": [
                                {
                                    "char_caption": "",
                                    "centers": [
                                        {
                                            "x": 0.5,
                                            "y": 0.5
                                        }
                                    ]
                                }
                            ]
                        },
                        "legacy_uc": False,
                        "use_coords": False,
                        "use_order": True
                    }
                }
            }
        )
    except httpx.TimeoutException as exc:
        print_log(image_id, f"NovelAI request timed out: {type(exc).__name__}")
        raise fastapi.HTTPException(status_code=504, detail="NovelAI request timed out") from exc
    except httpx.RequestError as exc:
        print_log(image_id, f"NovelAI connection failed: {type(exc).__name__}")
        raise fastapi.HTTPException(status_code=502, detail="Could not connect to NovelAI") from exc

    if generation_response.status_code not in (200, 201):
        try:
            error_data = generation_response.json()
            error_message = (
                error_data.get("message") or generation_response.text
                if isinstance(error_data, dict) else generation_response.text
            )
        except ValueError:
            error_message = generation_response.text
        print_log(image_id, f"NovelAI HTTP {generation_response.status_code}: {str(error_message)[:500]}")

        if generation_response.status_code in (401, 403):
            detail = "NovelAI authentication failed"
            status_code = 502
        elif generation_response.status_code == 429:
            detail = "NovelAI generation is temporarily limited"
            status_code = 503
        else:
            detail = "NovelAI Image Generation Failed"
            status_code = 502

        raise fastapi.HTTPException(
            status_code=status_code,
            detail=detail
        )

    try:
        image_bytes = extract_image_from_zip(zip_data=generation_response.content)
    except (zipfile.BadZipFile, ValueError, RuntimeError, EOFError, OSError) as exc:
        print_log(image_id, f"Invalid NovelAI image archive: {type(exc).__name__}")
        raise fastapi.HTTPException(status_code=502, detail="Invalid NovelAI image response") from exc

    return image_bytes
