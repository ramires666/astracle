"""
Numba-optional scan kernels for turning target builders.

This module isolates heavy loops so the public target module stays readable and
within file-size limits.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:
    njit = None
    NUMBA_AVAILABLE = False


def _window_kernel_scan_py(
    center_idx: np.ndarray,
    direction: np.ndarray,
    event_amp: np.ndarray,
    n_rows: int,
    radius_days: int,
    distance_power: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Python fallback for event-window assignment."""
    best_weight = np.zeros(int(n_rows), dtype=np.float64)
    best_dir = np.zeros(int(n_rows), dtype=np.int32)
    best_event = np.full(int(n_rows), -1, dtype=np.int32)

    for e in range(len(center_idx)):
        c = int(center_idx[e])
        d = int(direction[e])
        a = float(event_amp[e])

        i0 = max(0, c - int(radius_days))
        i1 = min(int(n_rows) - 1, c + int(radius_days))
        for i in range(i0, i1 + 1):
            dist = abs(i - c)
            base_w = max(0.0, 1.0 - (float(dist) / float(radius_days + 1)))
            w = (base_w ** float(distance_power)) * a
            if w > best_weight[i]:
                best_weight[i] = w
                best_dir[i] = d
                best_event[i] = int(e)

    return best_weight, best_dir, best_event


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _window_kernel_scan_numba(
        center_idx: np.ndarray,
        direction: np.ndarray,
        event_amp: np.ndarray,
        n_rows: int,
        radius_days: int,
        distance_power: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Numba JIT version of event-window assignment."""
        best_weight = np.zeros(n_rows, dtype=np.float64)
        best_dir = np.zeros(n_rows, dtype=np.int32)
        best_event = np.full(n_rows, -1, dtype=np.int32)

        for e in range(center_idx.shape[0]):
            c = int(center_idx[e])
            d = int(direction[e])
            a = float(event_amp[e])

            i0 = 0 if c - radius_days < 0 else c - radius_days
            i1 = (n_rows - 1) if c + radius_days >= n_rows else c + radius_days

            for i in range(i0, i1 + 1):
                dist = i - c
                if dist < 0:
                    dist = -dist

                base_w = 1.0 - (float(dist) / float(radius_days + 1))
                if base_w < 0.0:
                    base_w = 0.0
                w = (base_w ** distance_power) * a

                if w > best_weight[i]:
                    best_weight[i] = w
                    best_dir[i] = d
                    best_event[i] = e

        return best_weight, best_dir, best_event


def window_kernel_scan(
    center_idx: np.ndarray,
    direction: np.ndarray,
    event_amp: np.ndarray,
    n_rows: int,
    radius_days: int,
    distance_power: float,
    use_numba: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch event-window scan to Numba or Python implementation."""
    center_idx_arr = center_idx.astype(np.int64)
    direction_arr = direction.astype(np.int32)
    amp_arr = event_amp.astype(np.float64)

    if use_numba and NUMBA_AVAILABLE:
        return _window_kernel_scan_numba(
            center_idx=center_idx_arr,
            direction=direction_arr,
            event_amp=amp_arr,
            n_rows=int(n_rows),
            radius_days=int(radius_days),
            distance_power=float(distance_power),
        )

    return _window_kernel_scan_py(
        center_idx=center_idx_arr,
        direction=direction_arr,
        event_amp=amp_arr,
        n_rows=int(n_rows),
        radius_days=int(radius_days),
        distance_power=float(distance_power),
    )


def _segment_midpoint_scan_py(
    start_idx: np.ndarray,
    end_idx: np.ndarray,
    direction: np.ndarray,
    seg_amp: np.ndarray,
    n_rows: int,
    center_power: float,
    min_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Python fallback for midpoint-segment assignment."""
    best_weight = np.zeros(int(n_rows), dtype=np.float64)
    best_dir = np.zeros(int(n_rows), dtype=np.int32)
    best_segment = np.full(int(n_rows), -1, dtype=np.int32)

    for i in range(len(start_idx)):
        st = int(start_idx[i])
        en = int(end_idx[i])
        if en < st:
            st, en = en, st

        d = int(direction[i])
        a = float(seg_amp[i])

        m = (st + en) / 2.0
        half = max((en - st) / 2.0, 1.0)

        for j in range(st, en + 1):
            dist_norm = abs(float(j) - m) / half
            shape = max(0.0, 1.0 - dist_norm)
            w = (shape ** float(center_power)) * a
            if w < float(min_weight):
                w = float(min_weight)
            if w > best_weight[j]:
                best_weight[j] = w
                best_dir[j] = d
                best_segment[j] = i

    return best_weight, best_dir, best_segment


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _segment_midpoint_scan_numba(
        start_idx: np.ndarray,
        end_idx: np.ndarray,
        direction: np.ndarray,
        seg_amp: np.ndarray,
        n_rows: int,
        center_power: float,
        min_weight: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Numba JIT version of midpoint-segment assignment."""
        best_weight = np.zeros(n_rows, dtype=np.float64)
        best_dir = np.zeros(n_rows, dtype=np.int32)
        best_segment = np.full(n_rows, -1, dtype=np.int32)

        for i in range(start_idx.shape[0]):
            st = int(start_idx[i])
            en = int(end_idx[i])
            if en < st:
                t = st
                st = en
                en = t

            d = int(direction[i])
            a = float(seg_amp[i])

            m = 0.5 * float(st + en)
            half = 0.5 * float(en - st)
            if half < 1.0:
                half = 1.0

            for j in range(st, en + 1):
                dist_norm = abs(float(j) - m) / half
                shape = 1.0 - dist_norm
                if shape < 0.0:
                    shape = 0.0
                w = (shape ** center_power) * a
                if w < min_weight:
                    w = min_weight
                if w > best_weight[j]:
                    best_weight[j] = w
                    best_dir[j] = d
                    best_segment[j] = i

        return best_weight, best_dir, best_segment


def segment_midpoint_scan(
    start_idx: np.ndarray,
    end_idx: np.ndarray,
    direction: np.ndarray,
    seg_amp: np.ndarray,
    n_rows: int,
    center_power: float,
    min_weight: float,
    use_numba: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch midpoint-segment scan to Numba or Python implementation."""
    start_idx_arr = start_idx.astype(np.int64)
    end_idx_arr = end_idx.astype(np.int64)
    direction_arr = direction.astype(np.int32)
    amp_arr = seg_amp.astype(np.float64)

    if use_numba and NUMBA_AVAILABLE:
        return _segment_midpoint_scan_numba(
            start_idx=start_idx_arr,
            end_idx=end_idx_arr,
            direction=direction_arr,
            seg_amp=amp_arr,
            n_rows=int(n_rows),
            center_power=float(center_power),
            min_weight=float(min_weight),
        )

    return _segment_midpoint_scan_py(
        start_idx=start_idx_arr,
        end_idx=end_idx_arr,
        direction=direction_arr,
        seg_amp=amp_arr,
        n_rows=int(n_rows),
        center_power=float(center_power),
        min_weight=float(min_weight),
    )
