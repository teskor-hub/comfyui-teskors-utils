"""Batch file renaming / copying node for ComfyUI ("TS Rename Files In Dir").

The node takes every matching file in a directory and gives it a sequential
name of the form ``<prefix>_<zero padded index>_<original extension>``
(for example ``shot_0001_.png``).  Two modes are supported:

* **copy mode** - ``output_directory`` is set, sources are left untouched and
  renamed copies are written into the output directory;
* **in-place mode** - ``output_directory`` is empty, the files are renamed
  inside ``directory`` using a two phase (temp name -> final name) rename so
  that a new name can never collide with a not-yet-processed old name.

Because the operation is destructive, this module is deliberately defensive:

* the user supplied ``prefix`` is sanitised and every destination path is
  asserted to stay inside the target directory (a prefix such as
  ``../../evil`` or ``/tmp/x`` must never escape);
* index allocation is a single O(N) pass instead of rescanning the directory
  for every file;
* both rename phases roll back to the original names when anything fails;
* only whitelisted extensions are touched by default;
* ``dry_run`` lets the user inspect the planned mapping before committing.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import uuid
from typing import Iterable, Sequence

from ..ts_utils.filesort import SORT_METHODS, leading_int, sort_names

__all__ = ["RenameFilesInDir", "sort_by", "sort_methods", "extract_first_number"]


# Prefix used by phase 1 of the in-place rename.  Files carrying it are leftovers
# of an interrupted run and are never picked up again as rename candidates.
TEMP_PREFIX = "__tmp__"

# Characters that must never end up in a generated file name: path separators,
# the Windows drive separator and the usual reserved characters.
_UNSAFE_NAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')

# Empty means "accept every regular file", which is what this node did before the
# extension filter existed. Narrowing the default would silently turn the node into
# a no-op for video folders, which is its main use here.
DEFAULT_EXTENSIONS = ""


# Ordering lives in ts_utils.filesort so this node and the video-batch loader
# cannot drift apart. The aliases below keep this module's historical public
# names working for anything that imported them directly.
sort_methods = list(SORT_METHODS)
sort_by = sort_names
extract_first_number = leading_int


def _parse_extensions(extensions: str | None) -> set[str] | None:
    """Turn the ``extensions`` widget value into a set of lowercase suffixes.

    Returns ``None`` when every file should be accepted (empty widget).
    """
    if extensions is None:
        return None

    raw = str(extensions).strip()
    if not raw:
        return None

    allowed: set[str] = set()
    for part in re.split(r"[,;\s]+", raw):
        part = part.strip().lower()
        if not part:
            continue
        if not part.startswith("."):
            part = "." + part
        allowed.add(part)

    return allowed or None


def _safe_list_files(directory: str, allowed_ext: set[str] | None = None) -> list[str]:
    """List regular files in ``directory``, filtered by extension.

    Leftovers from an interrupted in-place run (``__tmp__*``) are always
    skipped so a failed run cannot be mangled a second time.
    """
    out: list[str] = []
    for f in os.listdir(directory):
        if f.startswith(TEMP_PREFIX):
            continue
        if not os.path.isfile(os.path.join(directory, f)):
            continue
        if allowed_ext is not None and os.path.splitext(f)[1].lower() not in allowed_ext:
            continue
        out.append(f)
    return out


def _sanitize_prefix(prefix: str | None) -> str:
    """Strip anything that could turn the prefix into a path.

    ``prefix`` is a free-form STRING widget that is concatenated into the new
    file name and then joined onto the target directory.  Separators, ``..``
    and absolute/drive-qualified values would let the rename escape the target
    directory (and, because nothing then matches the collision probe, collapse
    every file onto a single destination).  Everything dangerous is replaced by
    an underscore.
    """
    raw = "" if prefix is None else str(prefix)
    cleaned = _UNSAFE_NAME_CHARS.sub("_", raw).strip()
    # A leading dot would create hidden files / ".." style names.
    cleaned = cleaned.strip(".").strip()
    return cleaned


def _format_name(index: int, digits: int, prefix: str, ext: str) -> str:
    """Build the new file name.

    ``ext`` is expected to include the dot (".png"/".jpg"/".jpeg").
    The underscore after the number is ALWAYS present, then the extension
    verbatim, e.g. ``prefix_0001_.png``.
    """
    num = str(index).zfill(digits)
    left = f"{prefix}_" if prefix else ""
    return f"{left}{num}_{ext}"


def _assert_inside(directory: str, new_name: str) -> str:
    """Return the destination path, refusing anything outside ``directory``."""
    if new_name != os.path.basename(new_name) or new_name in ("", ".", ".."):
        raise ValueError(f"Refusing to use unsafe file name: {new_name!r}")

    new_path = os.path.join(directory, new_name)
    if os.path.dirname(os.path.abspath(new_path)) != os.path.abspath(directory):
        raise ValueError(
            f"Refusing to write outside the target directory: {new_path!r} is not inside {directory!r}"
        )
    return new_path


class _IndexAllocator:
    """Hands out free indices for ``<prefix>_<num>_<ext>`` names.

    The directory is scanned exactly once; afterwards the taken indices live in
    a set and a monotonically increasing cursor walks forward.  The previous
    implementation re-listed (and stat-ed) the whole directory for every probe
    and restarted from 1 for every file, which made a rename of N files cost
    roughly N^2/2 directory listings.
    """

    def __init__(self, directory: str, digits: int, prefix: str, skip: Sequence[str] = ()) -> None:
        self._digits = int(digits)
        self._prefix = prefix
        left = f"{prefix}_" if prefix else ""
        self._pattern = re.compile(rf"^{re.escape(left)}(\d+)_")
        self._taken: set[int] = set()
        self._cursor = 1

        skip_set = set(skip)
        try:
            entries = os.listdir(directory)
        except OSError:
            entries = []

        for name in entries:
            if name in skip_set or name.startswith(TEMP_PREFIX):
                continue
            idx = self._index_of(name)
            if idx is not None:
                self._taken.add(idx)

    def _index_of(self, name: str) -> int | None:
        """Index encoded in ``name``, or ``None`` if it is not one of ours."""
        m = self._pattern.match(name)
        if not m:
            return None
        digits_part = m.group(1)
        try:
            idx = int(digits_part)
        except ValueError:
            return None
        # Only accept the exact zero padding this node produces, so unrelated
        # files are not mistaken for reserved indices.
        if str(idx).zfill(self._digits) != digits_part:
            return None
        return idx

    def next_index(self) -> int:
        while self._cursor in self._taken:
            self._cursor += 1
        return self._cursor

    def reserve(self, index: int) -> None:
        self._taken.add(index)
        if index >= self._cursor:
            self._cursor = index + 1


def _directory_signature(directory: str) -> str:
    """Stable digest of a directory's contents (name + mtime + size)."""
    try:
        entries = sorted(os.listdir(directory))
    except OSError:
        return "missing"

    parts: list[str] = []
    for name in entries:
        path = os.path.join(directory, name)
        try:
            st = os.stat(path)
        except OSError:
            parts.append(f"{name}|?|?")
            continue
        parts.append(f"{name}|{st.st_mtime_ns}|{st.st_size}")

    return hashlib.sha256("\n".join(parts).encode("utf-8", "surrogatepass")).hexdigest()


class RenameFilesInDir:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "output_directory": ("STRING", {"default": ""}),
                "sort_method": (sort_methods,),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "files_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "prefix": ("STRING", {"default": ""}),
                "digits": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
                "extensions": ("STRING", {"default": DEFAULT_EXTENSIONS}),
                "dry_run": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "run_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("COUNT",)
    FUNCTION = "run"
    CATEGORY = "TS Utils/Files"

    @classmethod
    def IS_CHANGED(cls, **kwargs) -> float | str:
        """Only re-run when something actually changed.

        Returning NaN unconditionally made this destructive node re-rename the
        directory on every single Queue press, even for unrelated graph edits.
        The cache key is now derived from the widget values plus the real state
        of the directories involved; ``run_always`` restores the old behaviour
        for users who want it.
        """
        if kwargs.get("run_always"):
            return float("NaN")

        payload: list[str] = [f"{k}={kwargs[k]!r}" for k in sorted(kwargs) if k != "run_always"]

        for key in ("directory", "output_directory"):
            value = kwargs.get(key)
            if isinstance(value, str) and value.strip():
                payload.append(f"{key}:{_directory_signature(value)}")

        return hashlib.sha256("\n".join(payload).encode("utf-8", "surrogatepass")).hexdigest()

    def run(
        self,
        directory: str,
        output_directory: str = "",
        sort_method: str | None = None,
        start_index: int = 0,
        files_load_cap: int = 0,
        prefix: str = "",
        digits: int = 4,
        extensions: str = DEFAULT_EXTENSIONS,
        dry_run: bool = False,
        run_always: bool = False,
    ) -> tuple[int]:
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")

        safe_prefix = _sanitize_prefix(prefix)
        allowed_ext = _parse_extensions(extensions)

        files = _safe_list_files(directory, allowed_ext)
        if not files:
            return (0,)

        files = sort_by(files, directory, sort_method)
        files = files[start_index:]

        if files_load_cap > 0:
            files = files[:files_load_cap]

        if not files:
            return (0,)

        inplace = (output_directory is None) or (str(output_directory).strip() == "")

        if inplace:
            return self._run_inplace(directory, files, safe_prefix, digits, dry_run)

        return self._run_copy(directory, output_directory, files, safe_prefix, digits, dry_run)

    # ------------------------------------------------------------------
    # copy mode
    # ------------------------------------------------------------------
    def _run_copy(
        self,
        directory: str,
        output_directory: str,
        files: Sequence[str],
        prefix: str,
        digits: int,
        dry_run: bool,
    ) -> tuple[int]:
        output_directory = str(output_directory).strip()

        if os.path.abspath(output_directory) == os.path.abspath(directory):
            raise ValueError(
                "output_directory must differ from directory; leave it empty for an in-place rename."
            )

        if dry_run:
            allocator = _IndexAllocator(output_directory, digits, prefix)
            plan = self._build_plan(output_directory, files, prefix, digits, allocator)
            self._print_plan(plan, mode="copy (dry run)")
            return (len(plan),)

        os.makedirs(output_directory, exist_ok=True)
        allocator = _IndexAllocator(output_directory, digits, prefix)

        count = 0
        for fname in files:
            src = os.path.join(directory, fname)
            _, ext = os.path.splitext(fname)  # ext = ".png" / ".jpg" / ...

            idx = allocator.next_index()
            new_name = _format_name(idx, digits, prefix, ext)
            dst = _assert_inside(output_directory, new_name)

            if os.path.exists(dst):
                raise FileExistsError(f"Refusing to overwrite existing file: {dst}")

            shutil.copy2(src, dst)
            allocator.reserve(idx)
            count += 1

        return (count,)

    # ------------------------------------------------------------------
    # in-place mode
    # ------------------------------------------------------------------
    def _run_inplace(
        self,
        directory: str,
        files: Sequence[str],
        prefix: str,
        digits: int,
        dry_run: bool,
    ) -> tuple[int]:
        if dry_run:
            allocator = _IndexAllocator(directory, digits, prefix, skip=files)
            plan = self._build_plan(directory, files, prefix, digits, allocator)
            self._print_plan(plan, mode="in-place (dry run)")
            return (len(plan),)

        self._preflight(directory, files)

        used_temp: set[str] = set()

        def _make_temp_name(old_name: str) -> str:
            while True:
                t = f"{TEMP_PREFIX}{uuid.uuid4().hex}__{old_name}"
                if t not in used_temp and not os.path.exists(os.path.join(directory, t)):
                    used_temp.add(t)
                    return t

        # ---- phase 1: original -> temp -------------------------------
        temp_map: list[tuple[str, str]] = []  # (temp_name, original_name)
        try:
            for fname in files:
                old_path = os.path.join(directory, fname)
                tmp = _make_temp_name(fname)
                tmp_path = os.path.join(directory, tmp)

                os.rename(old_path, tmp_path)
                temp_map.append((tmp, fname))
        except OSError as exc:
            self._rollback(directory, temp_map)
            raise RuntimeError(
                f"Rename failed while moving files to temporary names (last file: {exc.filename!r}); "
                "all files were restored to their original names."
            ) from exc

        # Indices already used by files we are NOT touching.  At this point the
        # selected files carry temp names, so they cannot reserve an index.
        allocator = _IndexAllocator(directory, digits, prefix)

        # ---- phase 2: temp -> final ----------------------------------
        done: list[tuple[str, str, str]] = []  # (final_name, temp_name, original_name)
        try:
            for tmp, original_name in temp_map:
                tmp_path = os.path.join(directory, tmp)
                _, ext = os.path.splitext(original_name)

                idx = allocator.next_index()
                new_name = _format_name(idx, digits, prefix, ext)
                new_path = _assert_inside(directory, new_name)

                if os.path.exists(new_path):
                    raise FileExistsError(f"Refusing to overwrite existing file: {new_path}")

                os.rename(tmp_path, new_path)
                allocator.reserve(idx)
                done.append((new_name, tmp, original_name))
        except (OSError, ValueError) as exc:
            # Put the already renamed files back onto their temp names, then
            # restore every temp name to the original one.
            for final_name, tmp, _original in reversed(done):
                try:
                    os.rename(os.path.join(directory, final_name), os.path.join(directory, tmp))
                except OSError:
                    pass
            self._rollback(directory, temp_map)
            raise RuntimeError(
                f"Rename failed while assigning final names ({exc}); "
                "all files were restored to their original names."
            ) from exc

        return (len(done),)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _preflight(directory: str, files: Sequence[str]) -> None:
        """Fail before touching anything if the batch obviously cannot work."""
        if not os.access(directory, os.W_OK | os.X_OK):
            raise PermissionError(f"Directory '{directory}' is not writable.")

        for fname in files:
            path = os.path.join(directory, fname)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"File disappeared before renaming: {path}")
            if not os.access(path, os.W_OK):
                raise PermissionError(f"File is not writable: {path}")

    @staticmethod
    def _rollback(directory: str, temp_map: Sequence[tuple[str, str]]) -> None:
        """Move every ``__tmp__`` file recorded in ``temp_map`` back."""
        for tmp, original_name in reversed(list(temp_map)):
            tmp_path = os.path.join(directory, tmp)
            if not os.path.exists(tmp_path):
                continue
            original_path = os.path.join(directory, original_name)
            if os.path.exists(original_path):
                continue
            try:
                os.rename(tmp_path, original_path)
            except OSError:
                pass

    @staticmethod
    def _build_plan(
        target_dir: str,
        files: Sequence[str],
        prefix: str,
        digits: int,
        allocator: _IndexAllocator,
    ) -> list[tuple[str, str]]:
        plan: list[tuple[str, str]] = []
        for fname in files:
            _, ext = os.path.splitext(fname)
            idx = allocator.next_index()
            new_name = _format_name(idx, digits, prefix, ext)
            _assert_inside(target_dir, new_name)
            allocator.reserve(idx)
            plan.append((fname, new_name))
        return plan

    @staticmethod
    def _print_plan(plan: Sequence[tuple[str, str]], mode: str) -> None:
        print(f"[TS Rename Files In Dir] {mode}: {len(plan)} file(s), nothing written.")
        for old, new in plan:
            print(f"  {old} -> {new}")
