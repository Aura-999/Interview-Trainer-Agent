"""
watsonx_client.py — Thin wrapper around ibm-watsonx-ai SDK.

Token-minimisation strategies applied here:
  1. max_new_tokens capped per task type (questions < answers < feedback).
  2. Response cache checked BEFORE any API call — identical inputs return instantly.
  3. Repetition penalty reduces token loops (saves tokens + improves quality).
  4. No system-prompt padding — all prompts are pre-compressed by prompt_builder.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as Params
from ibm_watsonx_ai.wml_client_error import InvalidCredentialsError
from tenacity import RetryError, retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from rag_engine import get_cached_response, make_cache_key, set_cached_response

load_dotenv()

_PLACEHOLDER_KEYS = {"your_api_key_here", "", None}


class WatsonxError(Exception):
    """
    Raised when the watsonx service is unavailable, misconfigured, or a call fails.

    Replaces the old FastAPI HTTPException now that this code runs in-process inside
    Streamlit instead of behind a FastAPI server — status_code/detail are kept as
    attributes so any calling code that wants them still can, but str(e) alone gives
    a full, displayable message (e.g. for st.error()).
    """

    def __init__(self, detail: str, status_code: int = 503):
        self.detail = detail
        self.status_code = status_code
        super().__init__(detail)


def _credentials_configured() -> bool:
    """Return False if .env still has placeholder values. Re-reads .env every call so a
    file save is picked up without restarting the process."""
    load_dotenv(override=True)
    return (
        os.getenv("WATSONX_API_KEY") not in _PLACEHOLDER_KEYS
        and os.getenv("WATSONX_PROJECT_ID") not in {"your_project_id_here", "", None}
    )

# ── Per-task token budgets (tune down to conserve Lite quota) ────────────────
TOKEN_BUDGETS = {
    "questions": int(os.getenv("MAX_NEW_TOKENS_QUESTIONS", "400")),
    "answer":    int(os.getenv("MAX_NEW_TOKENS_ANSWER",    "300")),
    "feedback":  int(os.getenv("MAX_NEW_TOKENS_FEEDBACK",  "280")),
}

# ── Shared generation params (conservative defaults) ────────────────────────
BASE_PARAMS = {
    Params.TEMPERATURE:         0.7,
    Params.TOP_P:               0.9,
    Params.REPETITION_PENALTY:  1.15,   # discourages token loops
    Params.STOP_SEQUENCES:      ["\n\n---", "###END###"],
}


@lru_cache(maxsize=1)
def _get_api_client() -> APIClient:
    if not _credentials_configured():
        raise WatsonxError(
            status_code=503,
            detail=(
                "IBM watsonx credentials are not configured. "
                "Set WATSONX_API_KEY and WATSONX_PROJECT_ID in your .env file."
            ),
        )
    creds = Credentials(
        url=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
        api_key=os.getenv("WATSONX_API_KEY"),
    )
    try:
        return APIClient(creds)
    except InvalidCredentialsError as e:
        _get_api_client.cache_clear()   # don't cache a failed client
        raise WatsonxError(
            status_code=503,
            detail=f"IBM watsonx authentication failed: invalid API key. Check your .env. ({e})",
        ) from e


def _get_model(task: str) -> ModelInference:
    return ModelInference(
        model_id=os.getenv("GRANITE_MODEL_ID", "meta-llama/llama-3-3-70b-instruct"),
        api_client=_get_api_client(),
        project_id=os.getenv("WATSONX_PROJECT_ID"),
        params={**BASE_PARAMS, Params.MAX_NEW_TOKENS: TOKEN_BUDGETS[task]},
    )


# Only retry on transient errors — never on auth/config failures (retrying won't fix a bad key)
@retry(
    retry=retry_if_not_exception_type((WatsonxError, InvalidCredentialsError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _call_granite(prompt: str, task: str) -> str:
    """Raw Granite call with retry on transient errors only."""
    model = _get_model(task)
    return model.generate_text(prompt=prompt)


def generate(prompt: str, task: str, cache_key_parts: list[str] | None = None) -> str:
    """
    Generate text via Granite, with transparent disk caching.

    Args:
        prompt:          The fully-built prompt string.
        task:            One of "questions" | "answer" | "feedback".
        cache_key_parts: Stable strings that uniquely identify this generation.
                         If supplied AND a cached result exists, Granite is NOT called.
    """
    # 1. Cache lookup — skip API call entirely if we have a hit
    if cache_key_parts:
        key = make_cache_key(task, *cache_key_parts)
        cached = get_cached_response(key)
        if cached:
            return cached  # zero tokens consumed

    # 2. Call Granite — let WatsonxError propagate cleanly up to the caller
    try:
        result = _call_granite(prompt, task)
    except WatsonxError:
        raise
    except InvalidCredentialsError as e:
        _get_api_client.cache_clear()
        raise WatsonxError(
            status_code=503,
            detail=f"IBM watsonx authentication failed: {e}",
        ) from e
    except RetryError as e:
        raise WatsonxError(
            status_code=503,
            detail=f"IBM watsonx call failed after retries: {e.last_attempt.exception()}",
        ) from e
    except Exception as e:
        raise WatsonxError(
            status_code=503,
            detail=f"IBM watsonx generation error: {e}",
        ) from e

    # 3. Store result for future calls
    if cache_key_parts:
        set_cached_response(key, result)

    return result
