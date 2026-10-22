"""
review.py — Step 6: Manual verification, correction helpers, and DataFrame builders
for student details, register-style attendance matrix, long-format attendance records,
and flagged review reports.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from attendance_reader import AttendanceDetectionResult, CellReading
from text_extractor import StudentRecord


def build_students_dataframe(students: List[StudentRecord]) -> pd.DataFrame:
    """Create the student list DataFrame (`students.csv` schema)."""
    rows = [
        {
            "S_No": int(s.s_no),
            "Roll_No": str(s.roll_no),
            "Student_Name": str(s.student_name),
            "Father_Name": str(s.father_name),
        }
        for s in students
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["Roll_No"] = df["Roll_No"].astype(str)
    return df


def build_matrix_dataframe(
    students: List[StudentRecord],
    detection: AttendanceDetectionResult,
    overrides: Dict[Tuple[int, int], str] | None = None,
) -> pd.DataFrame:
    """
    Create the register-style attendance matrix DataFrame (`attendance_matrix.csv` schema)
    containing only recorded sessions with dates, where 'P' -> 'Present' and blank/other -> 'Absent'.
    """
    overrides = overrides or {}
    status_lookup: Dict[Tuple[int, int], str] = {}
    for r in detection.readings:
        key = (r.row_idx, r.session_idx)
        status_lookup[key] = overrides.get(key, r.status)

    rows = []
    for r_idx, s in enumerate(students):
        row_dict: Dict[str, object] = {
            "S_No": int(s.s_no),
            "Roll_No": str(s.roll_no),
            "Student_Name": str(s.student_name),
            "Father_Name": str(s.father_name),
        }
        present_cnt = 0
        absent_cnt = 0
        for s_idx, col_key in zip(
            detection.recorded_session_indices, detection.recorded_date_sessions
        ):
            st_val = status_lookup.get((r_idx, s_idx), "Absent")
            # Enforce binary Present / Absent per user rule
            normalized_val = "Present" if str(st_val).strip().lower() in ("present", "p") else "Absent"
            row_dict[col_key] = normalized_val
            if normalized_val == "Present":
                present_cnt += 1
            else:
                absent_cnt += 1

        total_sessions = max(1, len(detection.recorded_date_sessions))
        row_dict["Present_Count"] = present_cnt
        row_dict["Absent_Count"] = absent_cnt
        row_dict["Attendance_%"] = round(100.0 * present_cnt / total_sessions, 1)
        rows.append(row_dict)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Roll_No"] = df["Roll_No"].astype(str)
    return df


def build_long_attendance_dataframe(
    edited_students_df: pd.DataFrame,
    edited_matrix_df: pd.DataFrame,
    detection: AttendanceDetectionResult,
    verified_keys: set[Tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """
    Build the long-format attendance DataFrame (`attendance.csv` schema) from the
    user-reviewed student list and matrix DataFrames.
    """
    verified_keys = verified_keys or set()
    reading_meta: Dict[Tuple[int, int], CellReading] = {
        (r.row_idx, r.session_idx): r for r in detection.readings
    }

    records = []
    for r_idx in range(len(edited_matrix_df)):
        m_row = edited_matrix_df.iloc[r_idx]
        s_row = edited_students_df.iloc[r_idx] if r_idx < len(edited_students_df) else m_row

        s_no = int(s_row["S_No"])
        roll_no = str(s_row["Roll_No"]).strip()
        student_name = str(s_row["Student_Name"]).strip()
        father_name = str(s_row["Father_Name"]).strip()

        for s_idx, col_key in zip(
            detection.recorded_session_indices, detection.recorded_date_sessions
        ):
            raw_cell_val = str(m_row.get(col_key, "Absent")).strip()
            final_status = (
                "Present" if raw_cell_val.lower() in ("present", "p") else "Absent"
            )
            orig = reading_meta.get((r_idx, s_idx))
            date_str = orig.date if orig else col_key.split("_")[0]
            sess_str = orig.session_label if orig else col_key.split("_")[-1]
            was_flagged = bool(orig.review_required) if orig else False
            is_verified = ((r_idx, s_idx) in verified_keys) or not was_flagged

            records.append(
                {
                    "S_No": s_no,
                    "Roll_No": roll_no,
                    "Student_Name": student_name,
                    "Father_Name": father_name,
                    "Date": date_str,
                    "Session": sess_str,
                    "Date_Session": col_key,
                    "Status": final_status,
                    "Confidence": orig.confidence if orig else 1.0,
                    "Review_Flag": "Verified" if is_verified else "Review Required",
                }
            )

    df = pd.DataFrame(records)
    if not df.empty:
        df["Roll_No"] = df["Roll_No"].astype(str)
    return df


def build_review_report_dataframe(
    edited_students_df: pd.DataFrame,
    edited_matrix_df: pd.DataFrame,
    detection: AttendanceDetectionResult,
    verified_keys: set[Tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """
    Build `review_report.csv` containing all cells that required human verification,
    along with their row/column position, reason, initial suggestion, and final verified status.
    """
    verified_keys = verified_keys or set()
    rows = []
    for item in detection.flagged_readings:
        r_idx = item.row_idx
        col_key = item.date_session_key

        roll_no = item.roll_no
        student_name = item.student_name
        if r_idx < len(edited_students_df):
            roll_no = str(edited_students_df.iloc[r_idx]["Roll_No"])
            student_name = str(edited_students_df.iloc[r_idx]["Student_Name"])

        final_status = item.status
        if r_idx < len(edited_matrix_df) and col_key in edited_matrix_df.columns:
            val = str(edited_matrix_df.iloc[r_idx][col_key]).strip()
            final_status = "Present" if val.lower() in ("present", "p") else "Absent"

        rows.append(
            {
                "Row_No": item.s_no,
                "Column_Session": col_key,
                "Roll_No": str(roll_no),
                "Student_Name": student_name,
                "Detected_Mark": item.raw_mark,
                "Ink_Stroke_%": round(item.stroke_ratio * 100.0, 2),
                "Suggested_Status": item.status,
                "Verified_Status": final_status,
                "Review_Status": (
                    "Verified by User"
                    if (item.row_idx, item.session_idx) in verified_keys
                    else "Pending Review"
                ),
                "Reason": item.review_reason,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Roll_No"] = df["Roll_No"].astype(str)
    return df


def recompute_matrix_totals(
    matrix_df: pd.DataFrame, session_cols: List[str]
) -> pd.DataFrame:
    """Recalculate Present_Count, Absent_Count, and Attendance_% after manual table edits."""
    df = matrix_df.copy()
    if df.empty or not session_cols:
        return df

    for col in session_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: "Present"
                if str(v).strip().lower() in ("present", "p")
                else "Absent"
            )

    present_counts = (df[session_cols] == "Present").sum(axis=1)
    total_sess = len(session_cols)
    df["Present_Count"] = present_counts.astype(int)
    df["Absent_Count"] = (total_sess - present_counts).astype(int)
    df["Attendance_%"] = (100.0 * present_counts / max(1, total_sess)).round(1)
    return df
