"""Immutable tuning parameters for the pose smoothing pipeline.

Every value that used to live as a mutable module-level global in
``nodes/openpose_smoother.py`` now lives here as a field of :class:`SmoothConfig`.
The old design mutated ``globals()`` under a process-wide lock for the duration of
one node execution, which serialised every run and leaked state whenever anything
raised.  A frozen dataclass threaded explicitly through the call graph removes both
problems and makes the pipeline safe to run concurrently.

Field names are intentionally kept UPPERCASE and identical to the historical global
names so the port stays a mechanical ``NAME`` -> ``cfg.NAME`` substitution that can
be diffed against git history.

The defaults below are the single source of truth for the node's widget defaults:
``SmoothConfig()`` is exactly the configuration the ComfyUI node builds when every
widget is left untouched.

This module deliberately imports nothing but the standard library so the pure-logic
parts of the package can be unit tested without numpy, cv2 or torch installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Tuple

__all__ = ["SmoothConfig", "DEFAULT_CONFIG"]


@dataclass(frozen=True)
class SmoothConfig:
    """All tunables consumed by :func:`ts_utils.pipeline.smooth_KPS_json_obj`.

    The dataclass is frozen so a single instance can be shared freely across
    threads and so ``DEFAULT_CONFIG`` can safely be a module singleton.  Build a
    variant with :meth:`dataclasses.replace` or by passing keyword arguments to the
    constructor; never mutate an existing instance.

    Container-typed fields use ``tuple``/``frozenset`` rather than ``list``/``set``
    because mutable defaults are both illegal on a frozen dataclass and the exact
    kind of shared mutable state this refactor exists to delete.  All consumers only
    iterate or membership-test them, so behaviour is unchanged.
    """

    # --- Root-scale carry: re-project the last good pose when the torso vanishes.
    ROOTSCALE_CARRY_ENABLED: bool = True
    CARRY_MAX_FRAMES: int = 48
    CARRY_MIN_ANCHORS: int = 2
    CARRY_ANCHOR_JOINTS: Tuple[int, ...] = (0, 1, 2, 5, 3, 6, 4, 7)
    CARRY_CONF_GATE: float = 0.20

    # --- Main-subject selection.
    FILTER_EXTRA_PEOPLE: bool = True
    MAIN_PERSON_MODE: str = "longest_track"
    TRACK_MATCH_MIN_PX: float = 80.0
    # Match radius is max(TRACK_MATCH_MIN_PX, TRACK_MATCH_FACTOR * torso size).
    # This was 3.0, which on a 720px-wide canvas reaches nearly half the frame:
    # two people simply merged into one track and the "subject" teleported
    # between them. Measured against a synthetic ground truth, 3.0 and 2.0 both
    # merge (RMS 156 px); 1.5 and below keep them apart (RMS 10 px) with no
    # effect at all on single-subject footage. 1.5 is the loosest value that
    # works, leaving the most headroom for fast movement.
    TRACK_MATCH_FACTOR: float = 1.5
    TRACK_MAX_FRAME_GAP: int = 32
    TRACK_DIST_PENALTY: float = 1.5

    # --- Identity-switch guard (opt-in).
    # A detection may only join a track if their torso sizes are within this
    # ratio of each other. Off by default: tightening TRACK_MATCH_FACTOR already
    # separates people cleanly, and measurement showed this gate is what made
    # single-subject output diverge from the pre-1.0 releases. Turn it on for
    # crowded footage where people of very different sizes cross paths.
    # 0 disables the check.
    TRACK_SCALE_TOLERANCE: float = 0.0

    # --- Subject scoring weights (see tracking.subject_score).
    # The old rule was "whoever appears in the most frames wins", which loses to
    # a steadily-detected bystander whenever the real subject's detection
    # flickers - precisely the situation this node exists to clean up. Presence
    # still carries the most weight, but size (closeness to camera) and
    # centrality now get a say. Weights are relative; they need not sum to 1.
    SUBJECT_W_COVERAGE: float = 0.40
    SUBJECT_W_SCALE: float = 0.30
    SUBJECT_W_CENTER: float = 0.20
    SUBJECT_W_CONF: float = 0.10
    FACE_WEIGHT_IN_SCORE: float = 0.15
    HAND_WEIGHT_IN_SCORE: float = 0.35

    # --- Spatial outlier suppression.
    SPATIAL_OUTLIER_FIX: bool = True
    BONE_MAX_FACTOR: float = 2.3
    TORSO_RADIUS_FACTOR: float = 4.0

    # --- Per-frame velocity-damped EMA over the body skeleton.
    ALPHA_BODY: float = 0.70
    MAX_STEP_BODY: float = 60.0
    VEL_ALPHA: float = 0.45
    EPS: float = 0.3
    CONF_GATE_BODY: float = 0.35

    # --- Joints allowed to legitimately disappear (hands/elbows) and so never carried.
    ALLOW_DISAPPEAR_JOINTS: FrozenSet[int] = frozenset({3, 4, 6, 7})

    # --- Gap filling / short-run rejection on the body sequence.
    GAP_FILL_ENABLED: bool = True
    MAX_GAP_FRAMES: int = 12
    MIN_RUN_FRAMES: int = 3

    # --- Torso appearance synchronisation.
    TORSO_SYNC_ENABLED: bool = True
    TORSO_JOINTS: FrozenSet[int] = frozenset({1, 2, 5, 8, 11})
    TORSO_LOOKAHEAD_FRAMES: int = 32

    # --- Zero-lag (forward+backward) EMA over the whole body sequence.
    SUPER_SMOOTH_ENABLED: bool = True
    SUPER_SMOOTH_ALPHA: float = 0.7
    SUPER_SMOOTH_MIN_CONF: float = 0.20

    MEDIAN3_ENABLED: bool = True

    # --- Dense (face / hand) keypoint handling.
    FACE_SMOOTH_ENABLED: bool = True
    HANDS_SMOOTH_ENABLED: bool = False
    CONF_GATE_FACE: float = 0.20
    CONF_GATE_HAND: float = 0.60
    HAND_MIN_POINTS_PRESENT: int = 7
    MIN_HAND_RUN_FRAMES: int = 6
    DENSE_GAP_FILL_ENABLED: bool = False
    DENSE_MAX_GAP_FRAMES: int = 12
    DENSE_MIN_RUN_FRAMES: int = 3
    DENSE_MEDIAN3_ENABLED: bool = False
    DENSE_SUPER_SMOOTH_ALPHA: float = 0.7


DEFAULT_CONFIG = SmoothConfig()
"""Shared default configuration.

Safe as a module-level singleton precisely because :class:`SmoothConfig` is frozen.
"""
