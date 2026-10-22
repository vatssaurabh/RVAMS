"""
grid_detector.py — Morphological horizontal and vertical line extraction,
line-intersection grid fitting, and cell bounding-box calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from image_processor import PreprocessedImage


@dataclass
class LineModel:
    """Linear model for a table grid line: Horizontal (y = m*x + c) or Vertical (x = m*y + c)."""

    position_ref: float
    slope: float
    intercept: float
    span: int

    def eval_at(self, coord: float) -> float:
        return self.slope * coord + self.intercept


@dataclass
class TableGrid:
    """Detected table grid structure with horizontal row lines and vertical column lines."""

    h_lines: List[LineModel]
    v_lines: List[LineModel]
    grid_mask: np.ndarray
    num_student_rows: int
    num_session_cols: int
    header_row_Start_idx: int = 0
    student_row_start_idx: int = 2
    session_col_start_idx: int = 4

    def intersect(self, h_idx: int, v_idx: int) -> Tuple[int, int]:
        """Compute exact (x, y) intersection of horizontal line `h_idx` and vertical line `v_idx`."""
        hl = self.h_lines[h_idx]
        vl = self.v_lines[v_idx]
        denom = 1.0 - vl.slope * hl.slope
        if abs(denom) < 1e-6:
            x = vl.intercept
        else:
            x = (vl.slope * hl.intercept + vl.intercept) / denom
        y = hl.slope * x + hl.intercept
        return int(round(x)), int(round(y))

    def get_grid_cell_bbox(
        self, h_top_idx: int, h_bot_idx: int, v_left_idx: int, v_right_idx: int
    ) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) bounding box for any grid cell [h_top..h_bot, v_left..v_right]."""
        hl1 = self.h_lines[h_top_idx]
        hl2 = self.h_lines[h_bot_idx]
        vl1 = self.v_lines[v_left_idx]
        vl2 = self.v_lines[v_right_idx]

        ymid_approx = 0.5 * (hl1.eval_at(500.0) + hl2.eval_at(500.0))
        x1 = int(round(vl1.eval_at(ymid_approx)))
        x2 = int(round(vl2.eval_at(ymid_approx)))
        xmid = 0.5 * (x1 + x2)
        y1 = int(round(hl1.eval_at(xmid)))
        y2 = int(round(hl2.eval_at(xmid)))
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    def get_student_cell_bbox(
        self, student_row_idx: int, col_idx: int
    ) -> Tuple[int, int, int, int]:
        """
        Return (x1, y1, x2, y2) for a student row (0-indexed, 0..num_student_rows-1)
        and column (0=S.N., 1=Roll No., 2=Name, 3=Father's Name, 4+=Session columns).
        """
        r_h = self.student_row_start_idx + student_row_idx
        return self.get_grid_cell_bbox(r_h, r_h + 1, col_idx, col_idx + 1)

    def get_session_header_bbox(self, session_idx: int) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) for session header cell (1, 2, 3, ...)."""
        c_v = self.session_col_start_idx + session_idx
        return self.get_grid_cell_bbox(1, 2, c_v, c_v + 1)

    def get_date_header_bbox(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) spanning the date header bar above the session columns."""
        last_v = min(len(self.v_lines) - 1, self.session_col_start_idx + self.num_session_cols)
        return self.get_grid_cell_bbox(0, 1, self.session_col_start_idx, last_v)


def detect_table_grid(preprocessed: PreprocessedImage) -> TableGrid:
    """
    Step 4 — Detect horizontal and vertical table lines using morphology and fit
    robust linear models for all row and column boundaries.
    """
    binary_inv = preprocessed.binary_inv
    img_h, img_w = binary_inv.shape[:2]

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35))

    h_lines_img = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, h_kernel)
    v_lines_img = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, v_kernel)

    grid_mask = cv2.bitwise_or(
        cv2.dilate(h_lines_img, np.ones((3, 3), np.uint8), iterations=1),
        cv2.dilate(v_lines_img, np.ones((3, 3), np.uint8), iterations=1),
    )

    # 1. Fit Horizontal Lines
    num_h, h_labels, h_stats, _ = cv2.connectedComponentsWithStats(h_lines_img)
    raw_h: List[LineModel] = []
    min_h_width = int(0.39 * img_w)
    for i in range(1, num_h):
        x, y, w, h, area = h_stats[i]
        if w >= min_h_width and int(0.165 * img_h) < y < int(0.93 * img_h):
            ys, xs = np.where(h_labels == i)
            m, c = np.polyfit(xs, ys, 1)
            y_mid = float(m * (0.5 * img_w) + c)
            raw_h.append(LineModel(position_ref=y_mid, slope=float(m), intercept=float(c), span=int(w)))

    raw_h.sort(key=lambda lm: lm.position_ref)

    # Deduplicate horizontal lines that are within 12px vertically
    h_models: List[LineModel] = []
    for lm in raw_h:
        if not h_models or abs(lm.position_ref - h_models[-1].position_ref) > 12.0:
            h_models.append(lm)
        elif lm.span > h_models[-1].span:
            h_models[-1] = lm

    # Keep the main contiguous table cluster (adjacent lines separated by <= 34px)
    if len(h_models) > 5:
        clusters: List[List[LineModel]] = [[h_models[0]]]
        for lm in h_models[1:]:
            if (lm.position_ref - clusters[-1][-1].position_ref) <= 34.0:
                clusters[-1].append(lm)
            else:
                clusters.append([lm])
        h_models = max(clusters, key=len)

    # 2. Fit Vertical Lines using top-band peak seeding + slope line-integral maximization
    top_y1 = int(h_models[1].position_ref) if len(h_models) > 2 else int(0.19 * img_h)
    top_y2 = min(img_h, top_y1 + int(0.16 * img_h))
    profile_top = np.sum(v_lines_img[top_y1:top_y2, :] > 0, axis=0)

    v_seeds: List[Tuple[int, int]] = []
    for x in range(int(0.08 * img_w), int(0.98 * img_w)):
        if profile_top[x] > 35:
            if not v_seeds or (x - v_seeds[-1][0]) > 14:
                v_seeds.append((x, int(profile_top[x])))
            elif profile_top[x] > v_seeds[-1][1]:
                v_seeds[-1] = (x, int(profile_top[x]))

    y_bot_limit = int(h_models[-1].position_ref) if h_models else int(0.88 * img_h)
    ys_sample = np.arange(top_y1, y_bot_limit, 2)

    v_models: List[LineModel] = []
    for x0, _ in v_seeds:
        best_m, best_c, best_score = 0.0, float(x0), -1
        for m_try in np.linspace(-0.01, 0.036, 93):
            for dx0 in (-2, -1, 0, 1, 2):
                c_try = (x0 + dx0) - m_try * 250.0
                xs_sample = np.clip(
                    np.round(m_try * ys_sample + c_try).astype(int), 0, img_w - 1
                )
                score = int(np.sum(v_lines_img[ys_sample, xs_sample] > 0))
                if score > best_score:
                    best_score = score
                    best_m = float(m_try)
                    best_c = float(c_try)
        v_models.append(
            LineModel(
                position_ref=float(best_m * 250.0 + best_c),
                slope=best_m,
                intercept=best_c,
                span=best_score,
            )
        )

    # Regularize slope of the narrow session columns so adjacent session borders stay parallel
    if len(v_models) > 5:
        anchor_m = v_models[4].slope
        for idx in range(5, len(v_models)):
            # Smoothly blend with anchor slope to prevent drift into adjacent column
            max_allowed = anchor_m + 0.0012 * (idx - 4)
            if v_models[idx].slope > max_allowed:
                x_at_220 = v_models[idx].eval_at(220.0)
                new_m = max_allowed
                new_c = x_at_220 - new_m * 220.0
                v_models[idx].slope = float(new_m)
                v_models[idx].intercept = float(new_c)

    num_student_rows = max(0, len(h_models) - 3)
    num_session_cols = max(0, len(v_models) - 5)

    return TableGrid(
        h_lines=h_models,
        v_lines=v_models,
        grid_mask=grid_mask,
        num_student_rows=num_student_rows,
        num_session_cols=num_session_cols,
        header_row_Start_idx=0,
        student_row_start_idx=2,
        session_col_start_idx=4,
    )


def draw_grid_overlay(
    image_bgr: np.ndarray,
    grid: TableGrid,
    active_sessions: Optional[List[int]] = None,
    cell_statuses: Optional[Dict[Tuple[int, int], str]] = None,
) -> np.ndarray:
    """
    Render a visual overlay showing detected horizontal/vertical grid lines and
    color-coded attendance cell statuses (Green=Present, Red=Absent, Orange=Review Required).
    """
    vis = image_bgr.copy()
    h, w = vis.shape[:2]

    # Draw horizontal lines in cyan
    x_left = max(0, int(grid.v_lines[0].eval_at(220)) - 5) if grid.v_lines else 0
    x_right = min(w - 1, int(grid.v_lines[-1].eval_at(880)) + 5) if grid.v_lines else w - 1
    for hl in grid.h_lines:
        y1 = int(round(hl.eval_at(x_left)))
        y2 = int(round(hl.eval_at(x_right)))
        cv2.line(vis, (x_left, y1), (x_right, y2), (255, 200, 0), 1, cv2.LINE_AA)

    # Draw vertical lines in blue
    y_top = max(0, int(grid.h_lines[0].eval_at(400)) - 5) if grid.h_lines else 0
    y_bot = min(h - 1, int(grid.h_lines[-1].eval_at(400)) + 5) if grid.h_lines else h - 1
    for idx, vl in enumerate(grid.v_lines):
        y_start = y_top if idx >= 4 else int(grid.h_lines[1].eval_at(200))
        x1 = int(round(vl.eval_at(y_start)))
        x2 = int(round(vl.eval_at(y_bot)))
        cv2.line(vis, (x1, y_start), (x2, y_bot), (255, 120, 0), 1, cv2.LINE_AA)

    # Highlight attendance cells for recorded sessions
    if cell_statuses:
        for (r_idx, s_idx), status in cell_statuses.items():
            if active_sessions is not None and s_idx not in active_sessions:
                continue
            x1, y1, x2, y2 = grid.get_student_cell_bbox(r_idx, grid.session_col_start_idx + s_idx)
            if status == "Present":
                color = (40, 180, 40)  # Green
            elif status == "Review Required":
                color = (0, 140, 255)  # Orange
            else:
                color = (60, 60, 210)  # Red for Absent
            cv2.rectangle(vis, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), color, 2)

    return vis
