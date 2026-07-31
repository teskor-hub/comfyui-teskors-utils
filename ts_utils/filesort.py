"""Deterministic ordering for directory listings.

Both the video-batch loader and the renamer need the same thing: turn whatever
order the filesystem happened to hand back into a stable, predictable sequence.
Raw ``os.listdir`` order is arbitrary, so any node that assigns sequential
numbers must impose its own ordering first or it will scramble a sequence.

The public surface is :data:`SORT_METHODS` (the widget choices) and
:func:`sort_names`.

Implementation notes
--------------------
Ordering is expressed as a table of ``method -> (key builder, descending)``
rather than a branch per method, so adding a method touches one line and every
method is guaranteed to go through the same ``sorted`` call.  Keys are built
once per entry via :func:`sorted`'s ``key=`` argument, so ``stat`` is called at
most once per file even for the timestamp orders.
"""

from __future__ import annotations

import os
import re
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

__all__ = ["SORT_METHODS", "sort_names", "leading_int"]


#: Widget choices, in the order they appear in the dropdown. These exact strings
#: are persisted inside saved user workflows, so they must never be reworded.
SORT_METHODS: Sequence[str] = (
    "None",
    "Alphabetical (ASC)",
    "Alphabetical (DESC)",
    "Numerical (ASC)",
    "Numerical (DESC)",
    "Datetime (ASC)",
    "Datetime (DESC)",
)

_DIGITS = re.compile(r"[0-9]+")

#: Sorts after every real number, so unnumbered files cluster at the end of an
#: ascending numerical sort instead of being interleaved unpredictably.
_NO_NUMBER = float("inf")


def leading_int(name: str) -> float:
    """Return the first run of digits in ``name`` as an int.

    Falls back to :data:`_NO_NUMBER` when the name carries no digits at all.
    The file extension is stripped first so that ``clip.mp4`` does not sort as
    if it were numbered 4.
    """
    stem = os.path.splitext(name)[0]
    found = _DIGITS.search(stem)
    return int(found.group(0)) if found else _NO_NUMBER


def _mtime_in(directory: str) -> Callable[[str], float]:
    """Build a modification-time key bound to ``directory``.

    Unreadable entries sort first rather than raising: a single permission error
    in a folder should not take down a whole batch load.
    """

    def key(name: str) -> float:
        try:
            return os.path.getmtime(os.path.join(directory, name))
        except OSError:
            return float("-inf")

    return key


# method -> (key factory taking the directory, descending?)
_ORDERINGS: dict = {
    "Alphabetical (ASC)": (lambda _d: (lambda n: n), False),
    "Alphabetical (DESC)": (lambda _d: (lambda n: n), True),
    "Numerical (ASC)": (lambda _d: leading_int, False),
    "Numerical (DESC)": (lambda _d: leading_int, True),
    "Datetime (ASC)": (_mtime_in, False),
    "Datetime (DESC)": (_mtime_in, True),
}


def sort_names(
    names: Iterable[str],
    directory: str = ".",
    method: Optional[str] = None,
) -> List[str]:
    """Return ``names`` ordered by ``method``.

    ``method`` of ``None`` or ``"None"`` means "no particular order requested",
    which is still resolved to plain alphabetical order. Returning filesystem
    order here would make the renamer non-deterministic between runs on the same
    folder, which is far worse than picking an arbitrary but stable order.

    Ties are broken alphabetically so that equal timestamps or equal leading
    numbers still produce a total order.
    """
    entries = list(names)
    ordering: Optional[Tuple[Callable[[str], Callable[[str], object]], bool]]
    ordering = _ORDERINGS.get(method or "None")
    if ordering is None:
        return sorted(entries)

    build_key, descending = ordering
    primary = build_key(directory)
    return sorted(entries, key=lambda n: (primary(n), n), reverse=descending)
