"""
test_pipeline.py — End-to-end verification of the Attendance Sheet Digitization System
against the Section 7 Testing Checklist on the sample input sheet (`input_images/attendance.jpg`).
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from attendance_reader import detect_attendance_marks
from export_data import export_all_reports
from grid_detector import detect_table_grid
from image_processor import preprocess_image
from review import (
    build_long_attendance_dataframe,
    build_matrix_dataframe,
    build_review_report_dataframe,
    build_students_dataframe,
)
from text_extractor import extract_students_and_metadata


def run_verification() -> None:
    sample_path = os.path.join(PROJECT_ROOT, "input_images", "attendance.jpg")
    output_dir = os.path.join(PROJECT_ROOT, "output")
    preprocessed_out = os.path.join(output_dir, "preprocessed.jpg")

    print("1. Preprocessing sample image...")
    pre = preprocess_image(sample_path, output_save_path=preprocessed_out)
    assert os.path.exists(preprocessed_out), "preprocessed.jpg was not saved"
    h, w = pre.rectified_bgr.shape[:2]
    assert h > w, "Image should be in portrait orientation"

    print("2. Detecting table grid...")
    grid = detect_table_grid(pre)
    print(
        f"   Detected {grid.num_student_rows} student rows and {grid.num_session_cols} session columns."
    )
    assert grid.num_student_rows == 28, f"Expected 28 student rows, got {grid.num_student_rows}"
    assert grid.num_session_cols == 6, f"Expected 6 session columns, got {grid.num_session_cols}"

    print("3. Extracting printed student details and date/session headers via OCR...")
    students, metadata = extract_students_and_metadata(pre, grid)
    assert len(students) == 28, f"Expected 28 StudentRecord entries, got {len(students)}"
    assert students[0].roll_no == "250851002001", f"Unexpected first roll_no: {students[0].roll_no}"
    assert students[-1].roll_no == "250851002031", f"Unexpected last roll_no: {students[-1].roll_no}"
    assert metadata.sheet_date == "2026-08-13", f"Expected date 2026-08-13, got {metadata.sheet_date}"

    print("4. Detecting attendance marks (recorded sessions only)...")
    detection = detect_attendance_marks(pre, grid, students, metadata)
    print(f"   Recorded sessions: {detection.recorded_date_sessions}")
    assert detection.recorded_date_sessions == [
        "2026-08-13_S1",
        "2026-08-13_S2",
        "2026-08-13_S3",
    ], f"Expected only S1..S3 as recorded sessions, got {detection.recorded_date_sessions}"

    # Verify Row 15 (Deepanshu Kumar, 250851002016) Session 3 '..' mark is flagged for review
    # and defaulted to 'Absent' per the rule: 'P' = Present, blank/other = Absent
    flagged_keys = {(f.s_no, f.date_session_key): f for f in detection.flagged_readings}
    assert (15, "2026-08-13_S3") in flagged_keys, (
        "Row 15 (Deepanshu Kumar) S3 uncertain '..' mark should be flagged for review"
    )
    row15_s3 = flagged_keys[(15, "2026-08-13_S3")]
    assert row15_s3.status == "Absent", "Non-'P' uncertain mark should default to Absent"

    print("5. Building DataFrames and exporting CSV/Excel reports...")
    students_df = build_students_dataframe(students)
    matrix_df = build_matrix_dataframe(students, detection)
    long_df = build_long_attendance_dataframe(students_df, matrix_df, detection)
    review_df = build_review_report_dataframe(students_df, matrix_df, detection)

    export_res = export_all_reports(
        students_df=students_df,
        attendance_long_df=long_df,
        attendance_matrix_df=matrix_df,
        review_report_df=review_df,
        output_dir=output_dir,
    )

    for fname, fpath in export_res.file_paths.items():
        assert os.path.exists(fpath), f"Expected output file missing: {fpath}"
        print(f"   [OK] Generated {fname} ({os.path.getsize(fpath)} bytes)")

    # Verify CSV roll numbers remain exact strings when read back
    check_df = pd.read_csv(export_res.file_paths["students.csv"], dtype={"Roll_No": str})
    assert check_df.iloc[0]["Roll_No"] == "250851002001"
    assert len(check_df) == 28

    print("\n--- Matrix Preview (First 10 Rows) ---")
    print(matrix_df.head(10).to_string(index=False))
    print("\n--- Flagged Review Report ---")
    print(review_df.to_string(index=False))
    print("\nALL CHECKLIST TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_verification()
