"""Observational Ollama context diagnostics (no inference or routing limits).

A completed response can provide Ollama's actual prompt/eval counts. An interrupted
LangChain run may only provide callback estimates; those are labeled explicitly.
Ollama /api/ps exposes the allocated context of *currently loaded* models; an
unloaded model's effective context is deliberately reported as unknown.
"""

from __future__ import annotations

import logging
import threading
from contextvars import ContextVar
from typing import Any

import ollama
from langchain_core.callbacks import BaseCallbackHandler
from token_utils import count_tokens

LOGGER = logging.getLogger(__name__)
_CURRENT_MONITOR: ContextVar["ContextMonitor | None"] = ContextVar(
    "ollama_context_monitor", default=None
)


def get_monitor():
    return _CURRENT_MONITOR.get()


def set_monitor(monitor):
    return _CURRENT_MONITOR.set(monitor)


def reset_monitor(token):
    _CURRENT_MONITOR.reset(token)


def _field(value: Any, key: str, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def record_ollama_completion(model: str, response):
    """Consume completion counts from Ollama dicts or typed ChatResponse objects."""
    monitor = get_monitor()
    if monitor is not None:
        monitor.complete(
            model,
            _field(response, "prompt_eval_count"),
            _field(response, "eval_count"),
        )


def _model_key(value):
    return str(value or "").split(":", 1)[0].casefold()


def _token_count(value):
    if isinstance(value, str):
        return count_tokens(value)
    return count_tokens(str(value))


class ContextMonitor:
    def __init__(self):
        self._lock = threading.Lock()
        self.windows: dict[str, int] = {}
        self.samples: list[dict] = []
        self.active: dict[str, dict] = {}
        self.ollama_reachable: bool | None = None

    def refresh_windows(self):
        """Keep previously observed windows if a different model evicts a runner."""
        try:
            result = ollama.Client(timeout=3).ps()
            loaded = _field(result, "models", []) or []
            with self._lock:
                self.ollama_reachable = True
                for item in loaded:
                    name = _field(item, "name") or _field(item, "model")
                    length = _field(item, "context_length")
                    if name and isinstance(length, int) and length > 0:
                        self.windows[_model_key(name)] = length
            return True
        except Exception as error:
            LOGGER.debug("Ollama runtime context lookup unavailable: %s", error)
            with self._lock:
                self.ollama_reachable = False
            return False

    def window(self, model: str):
        with self._lock:
            return self.windows.get(_model_key(model))

    def start(self, run_id, model: str, approx_prompt: int):
        with self._lock:
            self.active[str(run_id)] = {
                "model": model, "prompt": approx_prompt, "generated": 0
            }

    def token(self, run_id, token):
        if not token:
            return
        with self._lock:
            sample = self.active.get(str(run_id))
            if sample is not None:
                sample["generated"] += _token_count(token)

    def complete(self, model: str, prompt_tokens, generated_tokens, run_id=None):
        try:
            prompt = int(prompt_tokens)
            generated = int(generated_tokens)
        except (TypeError, ValueError):
            return
        if min(prompt, generated) < 0:
            return
        with self._lock:
            if run_id is not None:
                self.active.pop(str(run_id), None)
            self.samples.append({
                "model": model, "prompt": prompt,
                "generated": generated, "estimated": False
            })

    def warning(self):
        """Report context pressure only with evidence; never equate any timeout with overflow."""
        with self._lock:
            samples = list(self.samples) + [
                {**sample, "estimated": True} for sample in self.active.values()
            ]
            windows = self.windows.copy()
        candidates = []
        for sample in samples:
            window = windows.get(_model_key(sample["model"]))
            total = sample["prompt"] + sample["generated"]
            if window and total >= window:
                candidates.append((total / window, sample, window, total))
        if not candidates:
            return None
        _, sample, window, total = max(candidates, key=lambda entry: entry[0])
        estimated = sample["estimated"]
        return {
            "code": "CONTEXT_WINDOW_PRESSURE",
            "model": sample["model"],
            "context_length": window,
            "tokens_used": total,
            "estimated": estimated,
            "message": (
                "Estimated token usage reached the configured Ollama context window. "
                "Earlier context may have been discarded."
                if estimated else
                "Ollama's reported prompt and generation token counts reached the "
                "configured context window. Context shifting or truncation may occur."
            ),
        }


class RetrievalContextCallback(BaseCallbackHandler):
    """Observe a single LangChain retrieval invocation without changing its behavior."""

    def __init__(self, monitor: ContextMonitor, model: str):
        self.monitor = monitor
        self.model = model

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        # The model's tool schema and chat templates add additional tokens, so
        # this prompt estimate is intentionally not represented as exact.
        text = "\n".join(
            str(getattr(message, "content", ""))
            for conversation in messages for message in conversation
        )
        self.monitor.start(run_id, self.model, _token_count(text))

    def on_llm_new_token(self, token, *, run_id, **kwargs):
        self.monitor.token(run_id, token)

    def on_llm_end(self, response, *, run_id, **kwargs):
        try:
            message = response.generations[0][0].message
            metadata = getattr(message, "response_metadata", {}) or {}
            usage = getattr(message, "usage_metadata", {}) or {}
            prompt = metadata.get("prompt_eval_count", usage.get("input_tokens"))
            generated = metadata.get("eval_count", usage.get("output_tokens"))
            self.monitor.complete(self.model, prompt, generated, run_id=run_id)
        except (IndexError, AttributeError, TypeError, KeyError):
            return

    def on_llm_error(self, error, *, run_id, **kwargs):
        # Keep partial estimates for the timeout diagnostic.
        pass
