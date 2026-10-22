"""
attendance_reader.py — Step 5: Detect handwritten attendance marks ('P') in grid cells,
filter to only recorded sessions with dates, classify 'P' as Present and blank/other
marks as Absent, and flag ambiguous cells (e.g., stray dots or smudges) for manual review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from grid_detector import TableGrid
from image_processor import PreprocessedImage
from text_extractor import ExtractedSheetMetadata, StudentRecord


@dataclass
class CellReading:
    """Detected attendance reading for a single (student_row, session_col) cell."""

    row_idx: int
    session_idx: int
    s_no: int
    roll_no: str
    student_name: str
    date: str
    session_label: str
    date_session_key: str
    status: str  # 'Present' or 'Absent'
    raw_mark: str  # 'P', 'Blank', or 'Uncertain (..)'
    stroke_ratio: float
    confidence: float
    review_required: bool
    review_reason: str
    bbox: Tuple[int, int, int, int]
    cell_crop_bgr: np.ndarray


@dataclass
class AttendanceDetectionResult:
    """Complete attendance detection result across all recorded sessions."""

    recorded_session_indices: List[int]
    recorded_date_sessions: List[str]
    readings: List[CellReading]
    flagged_readings: List[CellReading]


def _analyze_cell_ink(
    preprocessed: PreprocessedImage,
    grid: TableGrid,
    bbox: Tuple[int, int, int, int],
) -> Tuple[float, int, float]:
    """
    Crop the cell interior with an inward margin so grid borders are excluded,
    and measure:
      1. `stroke_ratio`: fraction of interior pixels with strong local pen-stroke response
      2. `num_components`: connected ink components count
      3. `vertical_span_ratio`: vertical height of the largest stroke relative to cell height
    """
    x1, y1, x2, y2 = bbox
    h_img, w_img = preprocessed.ink_response.shape[:2]

    # Inset margin to exclude horizontal & vertical grid lines
    pad_x = max(4, int(round(0.16 * (x2 - x1))))
    pad_y = max(4, int(round(0.16 * (y2 - y1))))

    ix1 = np.clip(x1 + pad_x, 0, w_img - 1)
    ix2 = np.clip(x2 - pad_x, ix1 + 1, w_img)
    iy1 = np.clip(y1 + pad_y, 0, h_img - 1)
    iy2 = np.clip(y2 - pad_y, iy1 + 1, h_img)

    resp_crop = preprocessed.ink_response[iy1:iy2, ix1:ix2]
    gmask_crop = grid.grid_mask[iy1:iy2, ix1:ix2]

    if resp_crop.size == 0:
        return 0.0, 0, 0.0

    # Pen stroke mask: local dark stroke (>22 intensity drop vs 15x15 local background)
    # and not part of a morphological table grid line
    stroke_mask = ((resp_crop > 22.0) & (gmask_crop == 0)).astype(np.uint8) * 255
    stroke_ratio = float(np.mean(stroke_mask > 0))

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(stroke_mask)
    valid_components = 0
    max_comp_h = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        if area >= 3:
            valid_components += 1
            if ch > max_comp_h:
                max_comp_h = ch

    cell_h = max(1, iy2 - iy1)
    vertical_span_ratio = float(max_comp_h) / float(cell_h)
    return stroke_ratio, valid_components, vertical_span_ratio


def detect_attendance_marks(
    preprocessed: PreprocessedImage,
    grid: TableGrid,
    students: List[StudentRecord],
    metadata: ExtractedSheetMetadata,
    present_min_stroke_ratio: float = 0.12,
    ambiguous_min_stroke_ratio: float = 0.025,
) -> AttendanceDetectionResult:
    """
    Analyze all student-session cells:
      1. Identify recorded sessions (columns that contain recorded 'P' attendance marks).
      2. Within recorded sessions:
         - 'P' mark (`stroke_ratio >= present_min_stroke_ratio` and vertical stroke span) -> `Present`
         - Blank or any other mark -> `Absent`
         - Uncertain/faint marks (`ambiguous_min_stroke_ratio <= stroke_ratio < present_min_stroke_ratio`,
           such as stray dots '..' or partial strokes) -> status='Absent' (per rule: only 'P' is Present,
           blank/other is Absent) with `review_required=True` so staff can verify it on the review screen.
    """
    h_img, w_img = preprocessed.rectified_bgr.shape[:2]

    # First pass: compute raw metrics for all session columns to determine which sessions are recorded
    raw_metrics: Dict[Tuple[int, int], Tuple[float, int, float, Tuple[int, int, int, int]]] = {}
    recorded_session_indices: List[int] = []

    for s_idx in range(grid.num_session_cols):
        present_count_in_col = 0
        for r_idx in range(grid.num_student_rows):
            bbox = grid.get_student_cell_bbox(r_idx, grid.session_col_start_idx + s_idx)
            s_ratio, n_comp, v_span = _analyze_cell_ink(preprocessed, grid, bbox)
            raw_metrics[(r_idx, s_idx)] = (s_ratio, n_comp, v_span, bbox)
            if s_ratio >= present_min_stroke_ratio and v_span >= 0.45:
                present_count_in_col += 1

        # A session column is considered a recorded session if at least 2 students have a 'P' mark
        if present_count_in_col >= 2:
            recorded_session_indices.append(s_idx)

    # Fallback: if a completely blank sheet is uploaded, keep the first session so table isn't empty
    if not recorded_session_indices and grid.num_session_cols > 0:
        recorded_session_indices = [0]

    recorded_date_sessions: List[str] = []
    readings: List[CellReading] = []
    flagged_readings: List[CellReading] = []

    for s_idx in recorded_session_indices:
        s_label = (
            metadata.session_labels[s_idx]
            if s_idx < len(metadata.session_labels)
            else f"S{s_idx + 1}"
        )
        date_session_key = f"{metadata.sheet_date}_{s_label}"
        recorded_date_sessions.append(date_session_key)

        for r_idx in range(grid.num_student_rows):
            student = students[r_idx]
            s_ratio, n_comp, v_span, bbox = raw_metrics[(r_idx, s_idx)]
            x1, y1, x2, y2 = bbox

            # Visual crop with slight context border for the manual review UI
            cx1 = max(0, x1 - 4)
            cy1 = max(0, y1 - 4)
            cx2 = min(w_img, x2 + 4)
            cy2 = min(h_img, y2 + 4)
            cell_crop = preprocessed.rectified_bgr[cy1:cy2, cx1:cx2].copy()

            if s_ratio >= present_min_stroke_ratio and v_span >= 0.40:
                status = "Present"
                raw_mark = "P"
                # Check if overly dense smudge (>48%) that might be a crossed-out mark
                if s_ratio > 0.48:
                    confidence = 0.68
                    review_required = True
                    review_reason = (
                        f"Heavy ink density ({s_ratio*100:.1f}%) — verify 'P' is not crossed out"
                    )
                else:
                    confidence = min(0.99, 0.78 + 0.6 * s_ratio)
                    review_required = False
                    review_reason = ""
            elif s_ratio >= ambiguous_min_stroke_ratio:
                # Per user rule: 'P' means Present, blank or any other value means Absent.
                # Still flag the cell for human verification on the review screen!
                status = "Absent"
                raw_mark = "Uncertain (..)"
                confidence = 0.52
                review_required = True
                review_reason = (
                    f"Non-'P' or faint mark detected (ink={s_ratio*100:.1f}%) — "
                    f"defaulted to Absent; please verify"
                )
            else:
                status = "Absent"
                raw_mark = "Blank"
                confidence = 0.97
                review_required = False
                review_reason = ""

            reading = CellReading(
                row_idx=r_idx,
                session_idx=s_idx,
                s_no=student.s_no,
                roll_no=student.roll_no,
                student_name=student.student_name,
                date=metadata.sheet_date,
                session_label=s_label,
                date_session_key=date_session_key,
                status=status,
                raw_mark=raw_mark,
                stroke_ratio=round(s_ratio, 4),
                confidence=round(confidence, 2),
                review_required=review_required,
                review_reason=review_reason,
                bbox=bbox,
                cell_crop_bgr=cell_crop,
            )
            readings.append(reading)
            if review_required:
                flagged_readings.append(reading)

    return AttendanceDetectionResult(
        recorded_session_indices=recorded_session_indices,
        recorded_date_sessions=recorded_date_sessions,
        readings=readings,
        flagged_readings=flagged_readings,
    )
