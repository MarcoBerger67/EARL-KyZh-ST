from __future__ import annotations

import functools
from pathlib import Path

import numpy as np


def rewrite_audio_path(audio_path: str, prefix_from: str, prefix_to: str) -> str:
    if not prefix_from:
        return audio_path
    if audio_path.startswith(prefix_to):
        return audio_path
    if not audio_path.startswith(prefix_from):
        raise ValueError(
            f"Audio path '{audio_path}' does not start with expected prefix '{prefix_from}'."
        )
    return prefix_to + audio_path.removeprefix(prefix_from)


def _normalize_audio_array(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    audio = np.squeeze(audio)
    if audio.ndim == 0:
        audio = audio.reshape(1)
    elif audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:
            audio = audio.mean(axis=0)
        else:
            audio = audio.mean(axis=1)
    elif audio.ndim > 2:
        audio = audio.reshape(-1)
    return np.asarray(audio, dtype=np.float32)


def _resample_audio_np(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)
    duration = audio.shape[0] / float(source_rate)
    target_length = max(int(round(duration * float(target_rate))), 1)
    source_positions = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)
    return np.interp(target_positions, source_positions, audio).astype(np.float32, copy=False)


@functools.lru_cache(maxsize=32768)
def load_audio_array(audio_path: str, target_sampling_rate: int) -> np.ndarray:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    last_error: Exception | None = None
    try:
        import soundfile as sf

        audio, sampling_rate = sf.read(str(path), always_2d=False)
        audio = _normalize_audio_array(audio)
        return _resample_audio_np(audio, int(sampling_rate), int(target_sampling_rate))
    except Exception as exc:  # pragma: no cover
        last_error = exc

    try:
        import librosa

        audio, _ = librosa.load(str(path), sr=int(target_sampling_rate), mono=True)
        return _normalize_audio_array(audio)
    except Exception as exc:  # pragma: no cover
        last_error = exc

    raise RuntimeError(
        "Failed to load audio. Install `soundfile` or `librosa` in the runtime environment."
    ) from last_error
