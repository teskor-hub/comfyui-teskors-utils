"""Multi-person association across a video and main-subject selection.

OpenPose emits an unordered list of people per frame with no identity across
frames.  This module stitches those detections into tracks and picks the one track
that represents the video's actual subject.

CONTRACT: :attr:`_Track.frames` stores the caller's *original* person dictionaries
by reference - never copies.  :mod:`ts_utils.pipeline` relies on ``is`` identity to
find the chosen person again inside the untouched input frame, and copying here
would silently duplicate people in the output when ``filter_extra_people`` is off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import SmoothConfig
from .keypoints import _body_center_from_pose, _dist, _estimate_torso_scale, _reshape_keypoints_2d, _sum_conf

__all__ = [
    "_Track",
    "_track_match_threshold_from_pose",
    "_build_tracks_over_video",
    "_track_presence_score",
    "_pick_main_track",
    "_choose_single_person",
]

Point = Tuple[float, float]


@dataclass
class _Track:
    """One person followed across frames.

    Attributes:
        frames: frame index -> the original person dict from the input data.
        centers: frame index -> that person's body centre.
        last_t: most recent frame index this track was seen in.
        last_center: body centre at ``last_t``, used for the next association step.
        last_scale: torso size at ``last_t``; ``None`` until one is measurable.
            Used to reject associations that would swap the track onto a person
            of a very different size.
    """

    frames: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    centers: Dict[int, Point] = field(default_factory=dict)
    last_t: int = -1
    last_center: Point = (0.0, 0.0)
    last_scale: Optional[float] = None


def _track_match_threshold_from_pose(pose_arr: Optional[List[float]], *, cfg: SmoothConfig) -> float:
    """Return how far a person may move between frames and still be the same person.

    Scaled by the subject's own torso size so that someone close to the camera is
    allowed to move more pixels per frame than someone far away, with a floor of
    ``cfg.TRACK_MATCH_MIN_PX``.
    """
    if isinstance(pose_arr, list):
        s = _estimate_torso_scale(_reshape_keypoints_2d(pose_arr))
        if s is not None:
            return max(float(cfg.TRACK_MATCH_MIN_PX), float(cfg.TRACK_MATCH_FACTOR) * float(s))
    return float(max(cfg.TRACK_MATCH_MIN_PX, 120.0))


def _pose_scale(pose_arr: Optional[List[float]]) -> Optional[float]:
    """Torso size of a person in pixels, or ``None`` if it cannot be measured."""
    if not isinstance(pose_arr, list):
        return None
    return _estimate_torso_scale(_reshape_keypoints_2d(pose_arr))


def _scale_compatible(track_scale: Optional[float], cand_scale: Optional[float], *, cfg: SmoothConfig) -> bool:
    """Whether a detection is plausibly the same person as a track, by size.

    Permissive by design: an unmeasurable size on either side is accepted rather
    than rejected, because a partially-occluded subject must not be dropped.  The
    check only fires when both sizes are known and clearly disagree.
    """
    tol = float(cfg.TRACK_SCALE_TOLERANCE)
    if tol <= 0:
        return True
    if track_scale is None or cand_scale is None:
        return True
    if track_scale <= 1e-6 or cand_scale <= 1e-6:
        return True
    ratio = track_scale / cand_scale
    if ratio < 1.0:
        ratio = 1.0 / ratio
    return ratio <= tol


def _build_tracks_over_video(frames_data: List[Any], *, cfg: SmoothConfig) -> List[_Track]:
    """Group per-frame detections into tracks by greedy nearest-centre association.

    Tracks are offered candidates most-recently-seen first, each takes its nearest
    unclaimed detection within the distance threshold, and any detection left over
    starts a new track.  A track that has not been seen for more than
    ``cfg.TRACK_MAX_FRAME_GAP`` frames stops competing, which lets a subject who
    walks out of frame and back in be re-acquired instead of stealing another
    person's detections.
    """
    tracks: List[_Track] = []
    for t, frame in enumerate(frames_data):
        if not isinstance(frame, dict):
            continue
        people = frame.get("people", [])
        if not isinstance(people, list) or not people:
            continue

        cand: List[Tuple[int, Dict[str, Any], Point, Optional[float]]] = []
        for i, p in enumerate(people):
            if not isinstance(p, dict):
                continue
            pose = p.get("pose_keypoints_2d")
            c = _body_center_from_pose(pose)
            if c is not None:
                cand.append((i, p, c, _pose_scale(pose)))
        if not cand:
            continue

        used = set()
        track_order = sorted(range(len(tracks)), key=lambda k: tracks[k].last_t, reverse=True)
        for k in track_order:
            tr = tracks[k]
            if (t - tr.last_t) > int(cfg.TRACK_MAX_FRAME_GAP):
                continue
            best_idx, best_d = None, 1e18
            for i, p, cc, sc in cand:
                if i in used:
                    continue
                if not _scale_compatible(tr.last_scale, sc, cfg=cfg):
                    continue
                thr = _track_match_threshold_from_pose(p.get("pose_keypoints_2d"), cfg=cfg)
                d = _dist(tr.last_center, cc)
                if d <= thr and d < best_d:
                    best_d = d
                    best_idx = i
            if best_idx is not None:
                i, p, cc, sc = next(x for x in cand if x[0] == best_idx)
                used.add(i)
                tr.frames[t], tr.centers[t], tr.last_t, tr.last_center = p, cc, t, cc
                if sc is not None:
                    tr.last_scale = sc
        for i, p, cc, sc in cand:
            if i not in used:
                tracks.append(_Track(frames={t: p}, centers={t: cc}, last_t=t, last_center=cc, last_scale=sc))
    return tracks


def _track_presence_score(tr: _Track) -> Tuple[int, float, float]:
    """Rank key for a track: ``(frames seen, total face conf, total body conf)``.

    Length dominates - the subject of a video is whoever is in most of it - with
    face and then body confidence breaking ties.
    """
    face_sum, body_sum = 0.0, 0.0
    for p in tr.frames.values():
        face_sum += _sum_conf(p.get("face_keypoints_2d"), 4)
        body_sum += _sum_conf(p.get("pose_keypoints_2d"), 1)
    return (len(tr.frames), face_sum, body_sum)


def _median(values: List[float]) -> float:
    """Median of ``values``; 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _track_stats(tr: _Track) -> Tuple[float, float, float]:
    """``(median torso scale, median centre, mean body confidence)`` for a track.

    The middle element is returned as a point-free scalar pair by the caller; here
    it is the median centre distance placeholder filled in by :func:`_pick_main_track`,
    which is the only place canvas geometry is known.
    """
    scales = [s for s in (_pose_scale(p.get("pose_keypoints_2d")) for p in tr.frames.values()) if s]
    confs = [_sum_conf(p.get("pose_keypoints_2d"), 1) for p in tr.frames.values()]
    return _median(scales), 0.0, (sum(confs) / len(confs) if confs else 0.0)


def _pick_main_track(
    tracks: List[_Track],
    *,
    cfg: SmoothConfig,
    n_frames: int = 0,
    canvas: Optional[Tuple[float, float]] = None,
) -> Optional[_Track]:
    """Return the track that best represents the video's subject.

    Scores each track on four normalised signals and takes the highest total:

    * **coverage** - share of frames the track appears in;
    * **scale** - median torso size relative to the largest track, i.e. who is
      closest to the camera;
    * **centrality** - how near the track sits to the middle of the canvas;
    * **confidence** - mean detection confidence.

    Ranking on frame count alone (the previous behaviour) hands the video to a
    background extra whenever the real subject's detection drops out, which is the
    common case in exactly the footage this node is meant to repair.  Weights come
    from ``cfg.SUBJECT_W_*``.

    With a single track the result is that track regardless of weights, so
    single-subject videos are unaffected.
    """
    if not tracks:
        return None
    if len(tracks) == 1:
        return tracks[0]

    total_frames = float(n_frames) if n_frames > 0 else float(max((len(t.frames) for t in tracks), default=1))
    total_frames = max(total_frames, 1.0)

    if canvas and canvas[0] > 0 and canvas[1] > 0:
        cx, cy = canvas[0] / 2.0, canvas[1] / 2.0
        max_offset = _dist((0.0, 0.0), (cx, cy)) or 1.0
    else:
        cx = cy = None
        max_offset = 1.0

    stats = [_track_stats(tr) for tr in tracks]
    max_scale = max((s[0] for s in stats), default=0.0) or 1.0
    max_conf = max((s[2] for s in stats), default=0.0) or 1.0

    best, best_score = None, -1e18
    for tr, (scale, _unused, conf) in zip(tracks, stats):
        coverage = len(tr.frames) / total_frames

        if cx is None:
            centrality = 0.0
        else:
            offsets = [_dist((cx, cy), c) for c in tr.centers.values()]
            centrality = max(0.0, 1.0 - (_median(offsets) / max_offset))

        score = (
            cfg.SUBJECT_W_COVERAGE * coverage
            + cfg.SUBJECT_W_SCALE * (scale / max_scale)
            + cfg.SUBJECT_W_CENTER * centrality
            + cfg.SUBJECT_W_CONF * (conf / max_conf)
        )
        if score > best_score:
            best_score, best = score, tr
    return best


def _choose_single_person(
    people: List[Dict[str, Any]], prev_center: Optional[Point], *, cfg: SmoothConfig
) -> Optional[Dict[str, Any]]:
    """Pick the most subject-like person in one frame.

    Scores each person by total body confidence plus weighted face and hand
    confidence, minus a penalty for distance from the previously chosen centre.  This
    is the per-frame fallback used when full-video tracking finds nothing; the
    returned dict is the caller's own object, not a copy.
    """
    if not people:
        return None
    best = None
    best_score = -1e18
    for p in people:
        pose = p.get("pose_keypoints_2d")
        score = _sum_conf(pose)
        score += cfg.FACE_WEIGHT_IN_SCORE * _sum_conf(p.get("face_keypoints_2d"), 4)
        score += cfg.HAND_WEIGHT_IN_SCORE * (
            _sum_conf(p.get("hand_left_keypoints_2d"), 2) + _sum_conf(p.get("hand_right_keypoints_2d"), 2)
        )
        center = _body_center_from_pose(pose)
        if prev_center is not None and center is not None:
            score -= cfg.TRACK_DIST_PENALTY * _dist(prev_center, center)
        if score > best_score:
            best_score = score
            best = p
    return best
