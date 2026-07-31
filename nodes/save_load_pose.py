"""Cache POSEDATA to disk so pose detection does not have to be re-run.

Detection is the slow part of a pose workflow. Saving the result lets you
iterate on generation without paying for ViTPose again, and lets you keep the
raw and smoothed versions side by side for comparison.

Files are ``.npz`` archives. The previous releases used pickle, which is an
arbitrary-code-execution format — the load node reads from ComfyUI's ``input``
folder, so a pose file shared by anyone else was untrusted input running with
your permissions. Those pickles also embedded the absolute install path of
ComfyUI-WanAnimatePreprocess as a module name, so they broke whenever the pack
moved. Neither is true of npz.
"""

from __future__ import annotations

import glob
import os
import time
from typing import List

import folder_paths

from ..ts_utils.posefile import POSE_FILE_EXT, load_posedata, save_posedata

__all__ = ["TSSavePoseDataAsPickle", "TSLoadPoseDataPickle"]


def _ensure_output_dir() -> str:
    out_dir = folder_paths.get_output_directory()
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _list_pose_files() -> List[str]:
    """Every pose archive under ComfyUI's input folder, recursively."""
    inp = folder_paths.get_input_directory()
    found = glob.glob(os.path.join(inp, "**", f"*{POSE_FILE_EXT}"), recursive=True)
    rel = sorted({os.path.relpath(f, inp).replace("\\", "/") for f in found if os.path.isfile(f)})
    return rel or [""]


def _abs_from_input(rel_path: str) -> str:
    return os.path.join(folder_paths.get_input_directory(), rel_path).replace("\\", "/")


def _unique_path(base_path: str) -> str:
    """``pose.npz`` -> ``pose_0001.npz`` -> ... so a save never overwrites."""
    if not os.path.exists(base_path):
        return base_path
    directory, name = os.path.split(base_path)
    base, ext = os.path.splitext(name)
    idx = 1
    while True:
        candidate = os.path.join(directory, f"{base}_{idx:04d}{ext}")
        if not os.path.exists(candidate):
            return candidate
        idx += 1


class TSSavePoseDataAsPickle:
    """Write POSEDATA to ComfyUI's output folder.

    The class name is kept for backwards compatibility: it is the value stored
    in every saved workflow that uses this node.
    """

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "filename": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = "TS Utils/Pose"

    def save(self, pose_data, filename: str = ""):
        out_dir = _ensure_output_dir()
        filename = (filename or "").strip()
        if not filename:
            filename = f"pose_data_{time.strftime('%Y%m%d_%H%M%S')}"
        # Accept a name the user typed with the old extension and correct it.
        for old in (".pkl", ".pickle", ".pt"):
            if filename.lower().endswith(old):
                filename = filename[: -len(old)]
        if not filename.lower().endswith(POSE_FILE_EXT):
            filename += POSE_FILE_EXT

        abs_path = _unique_path(os.path.join(out_dir, filename))
        save_posedata(pose_data, abs_path)
        # numpy appends .npz when the name lacks it; make sure we report reality.
        if not os.path.exists(abs_path) and os.path.exists(abs_path + ".npz"):
            abs_path += ".npz"
        return (abs_path,)


class TSLoadPoseDataPickle:
    """Read POSEDATA back from ComfyUI's input folder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file": (_list_pose_files(),),
            }
        }

    RETURN_TYPES = ("POSEDATA",)
    RETURN_NAMES = ("pose_data",)
    FUNCTION = "load"
    CATEGORY = "TS Utils/Pose"

    def load(self, file: str):
        if not isinstance(file, str) or not file.strip():
            raise ValueError(f"TS Load Pose Data: select a {POSE_FILE_EXT} file from the input folder.")

        abs_path = _abs_from_input(file)
        if not os.path.isfile(abs_path):
            raise ValueError(f"TS Load Pose Data: file not found: {abs_path}")

        if abs_path.lower().endswith((".pkl", ".pickle", ".pt")):
            raise ValueError(
                "TS Load Pose Data: .pkl files are no longer supported, because loading a pickle "
                "runs whatever code the file contains. Re-save your pose data with "
                "TS Save Pose Data to write a .npz instead. A converter for existing files is "
                "linked from the README."
            )

        return (load_posedata(abs_path),)
