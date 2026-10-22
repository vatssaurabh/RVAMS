"""
app.py — Streamlit web application for the Attendance Sheet Digitization System
(R.V. Higher Education & Technical Institute — Python Mini Project).
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, Tuple

import cv2
import pandas as pd
import streamlit as st

from attendance_reader import AttendanceDetectionResult, detect_attendance_marks
from export_data import export_all_reports
from grid_detector import TableGrid, detect_table_grid, draw_grid_overlay
from image_processor import PreprocessedImage, preprocess_image
from review import (
    build_long_attendance_dataframe,
    build_matrix_dataframe,
    build_review_report_dataframe,
    build_students_dataframe,
    recompute_matrix_totals,
)
from text_extractor import (
    ExtractedSheetMetadata,
    StudentRecord,
    extract_students_and_metadata,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAMPLE_IMAGE = os.path.join(BASE_DIR, "input_images", "attendance.jpg")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


st.set_page_config(
    page_title="Attendance Sheet Digitization System",
    page_icon="📋",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def run_digitization_pipeline(
    image_bytes: bytes,
    present_threshold: float,
) -> Tuple[
    PreprocessedImage,
    TableGrid,
    list[StudentRecord],
    ExtractedSheetMetadata,
    AttendanceDetectionResult,
]:
    """Run Steps 2–5 of the digitization pipeline and cache results by image hash."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    preprocessed_path = os.path.join(OUTPUT_DIR, "preprocessed.jpg")
    preprocessed = preprocess_image(
        image_bytes,
        output_save_path=preprocessed_path,
        apply_perspective_warp=False,
    )
    grid = detect_table_grid(preprocessed)
    students, metadata = extract_students_and_metadata(preprocessed, grid)
    detection = detect_attendance_marks(
        preprocessed,
        grid,
        students,
        metadata,
        present_min_stroke_ratio=present_threshold,
    )
    return preprocessed, grid, students, metadata, detection


def main() -> None:
    st.title("📋 Attendance Sheet Digitization System")
    st.caption(
        "R.V. Higher Education & Technical Institute — Convert photographed paper attendance "
        "registers into verified CSV & Excel records."
    )

    with st.sidebar:
        st.header("1. Input Attendance Sheet")
        source_mode = st.radio(
            "Select Image Source",
            options=["Use Sample Register Sheet", "Upload Photograph / Scan"],
            index=0,
        )

        image_bytes: bytes | None = None
        source_label = ""
        if source_mode == "Use Sample Register Sheet":
            if os.path.exists(DEFAULT_SAMPLE_IMAGE):
                with open(DEFAULT_SAMPLE_IMAGE, "rb") as f:
                    image_bytes = f.read()
                source_label = "input_images/attendance.jpg"
            else:
                st.error(f"Sample image not found at {DEFAULT_SAMPLE_IMAGE}")
        else:
            uploaded = st.file_uploader(
                "Upload Register Image (JPG / PNG)", type=["jpg", "jpeg", "png"]
            )
            if uploaded is not None:
                image_bytes = uploaded.getvalue()
                source_label = uploaded.name

        st.divider()
        st.subheader("Detection Settings")
        present_threshold = st.slider(
            "Minimum 'P' Ink Stroke Ratio",
            min_value=0.08,
            max_value=0.22,
            value=0.12,
            step=0.01,
            help=(
                "Cells in recorded sessions with stroke ratio above this threshold are "
                "marked Present ('P'). Blank or other marks are marked Absent."
            ),
        )
        st.info(
            "**Attendance Rule Active:**\n"
            "- Only **recorded sessions with dates** are included.\n"
            "- **`P`** $\\rightarrow$ **Present**\n"
            "- **Blank / Any other value** $\\rightarrow$ **Absent**\n"
            "- Ambiguous marks (e.g. `..`) are flagged for human confirmation."
        )

    if image_bytes is None:
        st.warning("Please upload an attendance sheet image or select the sample register sheet.")
        return

    img_hash = f"{hashlib.md5(image_bytes).hexdigest()}_{present_threshold:.2f}"

    with st.spinner("Running image preprocessing, grid detection, OCR, and mark analysis..."):
        preprocessed, grid, students, metadata, detection = run_digitization_pipeline(
            image_bytes, present_threshold
        )

    # Initialize session state when a new image or threshold is loaded
    if st.session_state.get("active_img_hash") != img_hash:
        st.session_state["active_img_hash"] = img_hash
        st.session_state["overrides"] = {}
        st.session_state["verified_keys"] = set()
        st.session_state["sheet_date"] = metadata.sheet_date
        st.session_state["students_df"] = build_students_dataframe(students)
        st.session_state["matrix_df"] = build_matrix_dataframe(
            students, detection, overrides={}
        )

    overrides: Dict[Tuple[int, int], str] = st.session_state["overrides"]
    verified_keys: set[Tuple[int, int]] = st.session_state["verified_keys"]
    students_df: pd.DataFrame = st.session_state["students_df"]
    matrix_df: pd.DataFrame = st.session_state["matrix_df"]

    # Compute live summary metrics
    session_cols = detection.recorded_date_sessions
    total_students = len(students_df)
    total_recorded_sessions = len(session_cols)
    total_cells = total_students * total_recorded_sessions
    present_cells = (
        int((matrix_df[session_cols] == "Present").sum().sum()) if session_cols else 0
    )
    overall_pct = round(100.0 * present_cells / max(1, total_cells), 1)
    pending_flags = sum(
        1
        for item in detection.flagged_readings
        if (item.row_idx, item.session_idx) not in verified_keys
    )

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    col_m1.metric("Students Extracted", f"{total_students}")
    col_m2.metric("Sheet Date", metadata.sheet_date)
    col_m3.metric("Recorded Sessions", f"{total_recorded_sessions} ({', '.join(metadata.session_labels[i] for i in detection.recorded_session_indices)})")
    col_m4.metric("Verified Present Rate", f"{overall_pct}% ({present_cells}/{total_cells})")
    col_m5.metric(
        "Cells Flagged for Review",
        f"{len(detection.flagged_readings)} ({pending_flags} pending)",
    )

    tab_img, tab_review, tab_export = st.tabs(
        [
            "🔍 1. Preprocess & Grid Detection",
            "✏️ 2. Manual Review & Register Matrix",
            "📥 3. Export CSV & Excel Reports",
        ]
    )

    # -------------------------------------------------------------------------
    # TAB 1: Image Preprocessing & Grid Detection
    # -------------------------------------------------------------------------
    with tab_img:
        st.subheader("Image Preprocessing & Table Grid Detection")
        st.markdown(
            f"**Source:** `{source_label}` &nbsp;|&nbsp; "
            f"**Detected Rows:** `{grid.num_student_rows}` students &nbsp;|&nbsp; "
            f"**Total Grid Session Columns:** `{grid.num_session_cols}` "
            f"(`{len(detection.recorded_session_indices)}` recorded with attendance marks)"
        )

        cell_status_map: Dict[Tuple[int, int], str] = {}
        for r in detection.readings:
            key = (r.row_idx, r.session_idx)
            if r.review_required and key not in verified_keys:
                cell_status_map[key] = "Review Required"
            else:
                cell_status_map[key] = overrides.get(key, r.status)

        overlay_bgr = draw_grid_overlay(
            preprocessed.rectified_bgr,
            grid,
            active_sessions=detection.recorded_session_indices,
            cell_statuses=cell_status_map,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Step 1 — Original Photograph**")
            st.image(
                cv2.cvtColor(preprocessed.original_bgr, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )
        with c2:
            st.markdown("**Step 2 — CLAHE Contrast Enhanced (`output/preprocessed.jpg`)**")
            st.image(preprocessed.enhanced_gray, use_container_width=True)
        with c3:
            st.markdown(
                "**Steps 4 & 5 — Detected Grid & Recorded Sessions**\n"
                "*(Green = Present, Red = Absent, Orange = Flagged for Review)*"
            )
            st.image(
                cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

    # -------------------------------------------------------------------------
    # TAB 2: Manual Review & Register Matrix
    # -------------------------------------------------------------------------
    with tab_review:
        st.subheader("Step 6 — Manual Verification & Correction")

        if detection.flagged_readings:
            st.warning(
                f"⚠️ **{len(detection.flagged_readings)} uncertain cell(s)** were detected with "
                "non-standard or faint ink strokes. Per the project rule, non-`P` marks default to "
                "**Absent**, but please inspect the cell crop below to confirm."
            )
            for idx, item in enumerate(detection.flagged_readings):
                key = (item.row_idx, item.session_idx)
                current_val = str(
                    matrix_df.iloc[item.row_idx].get(item.date_session_key, item.status)
                )
                with st.container(border=True):
                    fc1, fc2, fc3 = st.columns([1, 3, 2])
                    with fc1:
                        st.markdown("**Cropped Cell Zoom**")
                        crop_rgb = cv2.cvtColor(item.cell_crop_bgr, cv2.COLOR_BGR2RGB)
                        zoomed = cv2.resize(
                            crop_rgb, (0, 0), fx=3.5, fy=3.5, interpolation=cv2.INTER_NEAREST
                        )
                        st.image(zoomed, width=120)
                    with fc2:
                        st.markdown(
                            f"**Row #{item.s_no} — {item.student_name}** (`{item.roll_no}`)\n\n"
                            f"- **Session Column:** `{item.date_session_key}`\n"
                            f"- **Detected Raw Mark:** `{item.raw_mark}` "
                            f"(Interior ink stroke ratio: `{item.stroke_ratio*100:.1f}%`)\n"
                            f"- **Flag Reason:** {item.review_reason}"
                        )
                    with fc3:
                        new_choice = st.radio(
                            "Verified Attendance Status",
                            options=["Absent", "Present"],
                            index=0 if current_val == "Absent" else 1,
                            key=f"flag_radio_{item.row_idx}_{item.session_idx}",
                            horizontal=True,
                        )
                        if st.button(
                            "✅ Confirm Cell Status",
                            key=f"confirm_btn_{item.row_idx}_{item.session_idx}",
                            type="primary",
                        ):
                            overrides[key] = new_choice
                            verified_keys.add(key)
                            matrix_df.at[item.row_idx, item.date_session_key] = new_choice
                            st.session_state["matrix_df"] = recompute_matrix_totals(
                                matrix_df, session_cols
                            )
                            st.rerun()
        else:
            st.success("All cells in recorded sessions had high-confidence readings.")

        st.divider()
        st.markdown("#### Editable Register-Style Attendance Matrix (Recorded Sessions Only)")
        st.caption(
            "You can directly edit any student's Roll Number, Name, Father's Name, or toggle "
            "any session cell between **Present** and **Absent**."
        )

        col_cfg = {
            "S_No": st.column_config.NumberColumn("S.N.", disabled=True, width="small"),
            "Roll_No": st.column_config.TextColumn("Roll No. (Text)", width="medium"),
            "Student_Name": st.column_config.TextColumn("Student Name", width="medium"),
            "Father_Name": st.column_config.TextColumn("Father's Name", width="medium"),
            "Present_Count": st.column_config.NumberColumn("Present", disabled=True, width="small"),
            "Absent_Count": st.column_config.NumberColumn("Absent", disabled=True, width="small"),
            "Attendance_%": st.column_config.NumberColumn(
                "Attendance %", format="%.1f%%", disabled=True, width="small"
            ),
        }
        for scol in session_cols:
            col_cfg[scol] = st.column_config.SelectboxColumn(
                scol, options=["Present", "Absent"], required=True, width="small"
            )

        edited_matrix = st.data_editor(
            matrix_df,
            column_config=col_cfg,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            key="matrix_editor",
        )

        # Synchronize edits back to students_df and recompute row totals
        updated_matrix = recompute_matrix_totals(edited_matrix, session_cols)
        updated_students = updated_matrix[
            ["S_No", "Roll_No", "Student_Name", "Father_Name"]
        ].copy()
        st.session_state["matrix_df"] = updated_matrix
        st.session_state["students_df"] = updated_students
        matrix_df = updated_matrix
        students_df = updated_students

    # -------------------------------------------------------------------------
    # TAB 3: Export CSV & Excel Reports
    # -------------------------------------------------------------------------
    with tab_export:
        st.subheader("Steps 7 & 8 — Export Verified Data (CSV & Excel)")

        long_df = build_long_attendance_dataframe(
            students_df, matrix_df, detection, verified_keys=verified_keys
        )
        review_df = build_review_report_dataframe(
            students_df, matrix_df, detection, verified_keys=verified_keys
        )

        export_result = export_all_reports(
            students_df=students_df,
            attendance_long_df=long_df,
            attendance_matrix_df=matrix_df,
            review_report_df=review_df,
            output_dir=OUTPUT_DIR,
        )

        st.success(
            f"✅ All 5 output files have been automatically saved to `{OUTPUT_DIR}` and are ready for download."
        )

        dl1, dl2, dl3, dl4, dl5 = st.columns(5)
        with dl1:
            st.download_button(
                "⬇️ students.csv",
                data=students_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="students.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "⬇️ attendance_matrix.csv",
                data=matrix_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="attendance_matrix.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl3:
            st.download_button(
                "⬇️ attendance.csv (Long)",
                data=long_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="attendance.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl4:
            st.download_button(
                "⬇️ review_report.csv",
                data=review_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="review_report.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl5:
            st.download_button(
                "📊 attendance.xlsx",
                data=export_result.excel_bytes,
                file_name="attendance.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                type="primary",
            )

        st.divider()
        p_tab1, p_tab2, p_tab3, p_tab4 = st.tabs(
            [
                "📄 attendance_matrix.csv",
                "📄 students.csv",
                "📄 attendance.csv (Long Format)",
                "📄 review_report.csv",
            ]
        )
        with p_tab1:
            st.dataframe(matrix_df, use_container_width=True, hide_index=True)
        with p_tab2:
            st.dataframe(students_df, use_container_width=True, hide_index=True)
        with p_tab3:
            st.dataframe(long_df, use_container_width=True, hide_index=True)
        with p_tab4:
            st.dataframe(review_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
