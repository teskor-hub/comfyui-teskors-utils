"""Load every video in a directory as a list of ComfyUI IMAGE batches.

The node exists for chunked workflows: point it at a folder of clips, get one
IMAGE batch per clip plus the matching AUDIO, in a deterministic order.

Design notes
------------
Frames are decoded with OpenCV and rate conversion is done in the *index*
domain: a fractional stride ``source_fps / target_fps`` walks forward and the
nearest source frame is kept.  Working in frame indices rather than accumulated
timestamps keeps the output length exactly predictable
(``ceil(n_source / stride)``) and avoids drift on long clips.

Audio is decoded eagerly through ffmpeg into the dict shape ComfyUI expects
(``{"waveform": (1, channels, samples) float32, "sample_rate": int}``).  Eager
decoding is affordable here because the node already materialises every frame of
every clip in memory, so the audio is never the bottleneck.  When ffmpeg is not
installed the node degrades to silent audio instead of failing the whole graph.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover - exercised only on broken installs
    _HAS_CV2 = False

from ..ts_utils.filesort import SORT_METHODS, sort_names

__all__ = ["LoadVideoBatchListFromDir"]


#: Containers we are willing to hand to OpenCV.
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"})

#: Latent-space models want spatial dimensions on an 8 pixel grid; anything else
#: gets silently padded or rejected downstream.
SIZE_GRID = 8

#: Used when a file carries no audio track, or ffmpeg is unavailable.
FALLBACK_SAMPLE_RATE = 44100

#: Set this to point at a specific ffmpeg build; otherwise PATH is searched.
FFMPEG_ENV_VAR = "TS_FFMPEG_PATH"


# --------------------------------------------------------------------------- #
# external tools
# --------------------------------------------------------------------------- #


def _tool_path(name: str, env_var: Optional[str] = None) -> Optional[str]:
    """Locate an external binary, preferring an explicit override."""
    if env_var:
        override = os.environ.get(env_var)
        if override and os.path.isfile(override):
            return override
    return shutil.which(name)


def _run_capture(argv: Sequence[str]) -> Optional[bytes]:
    """Run ``argv`` and return stdout, or ``None`` if it failed in any way."""
    try:
        done = subprocess.run(argv, capture_output=True, check=False)
    except (OSError, ValueError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout


# --------------------------------------------------------------------------- #
# audio
# --------------------------------------------------------------------------- #


def _audio_layout(path: str) -> Optional[Tuple[int, int]]:
    """Return ``(sample_rate, channels)`` of the first audio stream.

    ``None`` means the probe failed or the file simply has no audio.
    """
    ffprobe = _tool_path("ffprobe")
    if ffprobe is None:
        return None

    raw = _run_capture(
        [
            ffprobe,
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels",
            "-of", "json",
            path,
        ]
    )
    if not raw:
        return None

    try:
        streams = json.loads(raw.decode("utf-8", "replace")).get("streams") or []
    except ValueError:
        return None
    if not streams:
        return None

    head = streams[0]
    try:
        return int(head["sample_rate"]), int(head["channels"])
    except (KeyError, TypeError, ValueError):
        return None


def _silent_audio(sample_rate: int = FALLBACK_SAMPLE_RATE) -> dict:
    """An empty but structurally valid AUDIO payload."""
    return {"waveform": torch.zeros((1, 1, 0), dtype=torch.float32), "sample_rate": sample_rate}


def _load_audio(path: str, start_s: float = 0.0, duration_s: Optional[float] = None) -> dict:
    """Decode a clip's audio into ComfyUI's AUDIO dict.

    Returns silence rather than raising when there is no audio track, no ffmpeg,
    or the decode fails: one clip without sound must not abort a batch load.
    """
    layout = _audio_layout(path)
    if layout is None:
        return _silent_audio()
    sample_rate, channels = layout
    if sample_rate <= 0 or channels <= 0:
        return _silent_audio()

    ffmpeg = _tool_path("ffmpeg", FFMPEG_ENV_VAR)
    if ffmpeg is None:
        return _silent_audio(sample_rate)

    argv: List[str] = [ffmpeg, "-v", "error"]
    if start_s > 0:
        argv += ["-ss", f"{start_s:.6f}"]
    argv += ["-i", path]
    if duration_s is not None and duration_s > 0:
        argv += ["-t", f"{duration_s:.6f}"]
    # 32-bit float PCM on stdout is the one format that needs no conversion on
    # our side: it maps straight onto a float32 tensor.
    argv += ["-vn", "-f", "f32le", "-acodec", "pcm_f32le", "-"]

    raw = _run_capture(argv)
    if not raw:
        return _silent_audio(sample_rate)

    flat = np.frombuffer(raw, dtype=np.float32)
    usable = (flat.size // channels) * channels
    if usable == 0:
        return _silent_audio(sample_rate)

    # ffmpeg writes interleaved samples; ComfyUI wants (batch, channel, sample).
    planar = flat[:usable].reshape(-1, channels).T.copy()
    return {
        "waveform": torch.from_numpy(planar).unsqueeze(0),
        "sample_rate": sample_rate,
    }


# --------------------------------------------------------------------------- #
# frames
# --------------------------------------------------------------------------- #


def _snap(value: int, grid: int = SIZE_GRID) -> int:
    """Round ``value`` to the nearest positive multiple of ``grid``."""
    if value <= 0:
        return 0
    return max(grid, int(round(value / grid)) * grid)


def _target_size(src_w: int, src_h: int, want_w: int, want_h: int) -> Tuple[int, int]:
    """Resolve the output resolution.

    A zero on either axis means "derive it from the other, keeping aspect";
    zero on both means "keep the source size".  The result is always snapped to
    :data:`SIZE_GRID`.
    """
    if want_w <= 0 and want_h <= 0:
        return _snap(src_w), _snap(src_h)
    if want_w <= 0:
        want_w = int(round(src_w * (want_h / float(src_h)))) if src_h else 0
    elif want_h <= 0:
        want_h = int(round(src_h * (want_w / float(src_w)))) if src_w else 0
    return _snap(want_w), _snap(want_h)


def _keep_indices(n_source: int, stride: float, every_nth: int, cap: int) -> List[int]:
    """Which source frame indices end up in the output, in order.

    ``stride`` is source-frames-per-output-frame (1.0 keeps everything).
    ``every_nth`` decimates on top of that, ``cap`` truncates (0 = no cap).
    """
    if n_source <= 0:
        return []
    step = stride if stride > 0 else 1.0
    picked: List[int] = []
    cursor = 0.0
    while True:
        idx = int(cursor)
        if idx >= n_source:
            break
        picked.append(idx)
        cursor += step
    if every_nth > 1:
        picked = picked[::every_nth]
    if cap > 0:
        picked = picked[:cap]
    return picked


def _decode_frames(
    path: str,
    *,
    force_rate: float,
    want_w: int,
    want_h: int,
    frame_load_cap: int,
    select_every_nth: int,
) -> Tuple[torch.Tensor, float, float]:
    """Decode one clip.

    Returns ``(frames, output_fps, duration_seconds)`` where ``frames`` is a
    ComfyUI IMAGE tensor of shape ``(N, H, W, 3)``, RGB, float32 in ``[0, 1]``.
    """
    if not _HAS_CV2:
        raise RuntimeError("OpenCV (opencv-python) is required to read video files.")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Cannot open video: {path}")

    try:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        n_source = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

        stride = 1.0
        if force_rate > 0 and source_fps > 0:
            stride = source_fps / float(force_rate)

        wanted = _keep_indices(n_source, stride, max(1, int(select_every_nth)), max(0, int(frame_load_cap)))
        wanted_set = set(wanted)
        out_w, out_h = _target_size(src_w, src_h, int(want_w), int(want_h))

        collected: List[np.ndarray] = []
        last_needed = wanted[-1] if wanted else -1
        index = 0
        while index <= last_needed:
            if not cap.grab():
                break
            if index in wanted_set:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    break
                if out_w > 0 and out_h > 0 and (frame.shape[1] != out_w or frame.shape[0] != out_h):
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                collected.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            index += 1
    finally:
        cap.release()

    if not collected:
        empty = torch.zeros((0, max(out_h, 1), max(out_w, 1), 3), dtype=torch.float32)
        return empty, 0.0, 0.0

    stacked = np.stack(collected, axis=0).astype(np.float32) / 255.0
    out_fps = (source_fps / stride / max(1, int(select_every_nth))) if source_fps > 0 else 0.0
    duration = (len(collected) / out_fps) if out_fps > 0 else 0.0
    return torch.from_numpy(stacked), out_fps, duration


# --------------------------------------------------------------------------- #
# node
# --------------------------------------------------------------------------- #


def _list_videos(directory: str) -> List[str]:
    """Every readable video file directly inside ``directory``."""
    found: List[str] = []
    for name in os.listdir(directory):
        if os.path.splitext(name)[1].lower() not in VIDEO_EXTENSIONS:
            continue
        if os.path.isfile(os.path.join(directory, name)):
            found.append(name)
    return found


class LoadVideoBatchListFromDir:
    """Load a folder of clips as one IMAGE batch (and AUDIO) per clip."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
                "force_rate": ("FLOAT", {"default": 0, "min": 0, "max": 120, "step": 1}),
                "width": ("INT", {"default": 720, "min": 0, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 1280, "min": 0, "max": 8192, "step": 1}),
            },
            "optional": {
                "video_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFF, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "load_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "sort_method": (list(SORT_METHODS),),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT")
    RETURN_NAMES = ("IMAGE", "audio", "COUNT")
    OUTPUT_IS_LIST = (True, True, False)

    FUNCTION = "load_videos"
    CATEGORY = "TS Utils/Video"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get("load_always"):
            return float("NaN")
        return hash(frozenset(kwargs.items()))

    def load_videos(
        self,
        directory: str,
        force_rate: float = 0,
        width: int = 0,
        height: int = 0,
        video_load_cap: int = 0,
        frame_load_cap: int = 0,
        select_every_nth: int = 1,
        start_index: int = 0,
        load_always: bool = False,
        sort_method: Optional[str] = None,
    ):
        if not directory or not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")

        names = _list_videos(directory)
        if not names:
            raise FileNotFoundError(
                f"No video files in '{directory}' (looking for: {', '.join(sorted(VIDEO_EXTENSIONS))})."
            )

        names = sort_names(names, directory, sort_method)[int(start_index) :]
        if video_load_cap > 0:
            names = names[: int(video_load_cap)]

        images: List[torch.Tensor] = []
        audios: List[dict] = []
        for name in names:
            path = os.path.join(directory, name)
            frames, _fps, duration = _decode_frames(
                path,
                force_rate=float(force_rate),
                want_w=int(width),
                want_h=int(height),
                frame_load_cap=int(frame_load_cap),
                select_every_nth=int(select_every_nth),
            )
            images.append(frames)
            audios.append(_load_audio(path, 0.0, duration or None))

        return (images, audios, len(images))
