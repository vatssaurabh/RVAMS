# Attendance Sheet Digitization System (`D:\ams`)

**R.V. Higher Education & Technical Institute — Python Mini Project**

Convert a photographed paper attendance register into structured, human-verified digital records (`CSV` and `Excel`) using Python, OpenCV image processing, OCR, and an interactive **Streamlit** web application.

---

## 1. Project Folder Structure

```text
D:\ams\
├── app.py                  # Streamlit web application (Upload, Preview, Review, Export)
├── image_processor.py      # Orientation normalization, CLAHE contrast & local ink response
├── grid_detector.py        # Morphological horizontal/vertical line & cell intersection detector
├── text_extractor.py       # OCR engine (PaddleOCR + Windows Native OCR) & text sanitization
├── attendance_reader.py    # Recorded session filter & 'P' vs. Blank/Absent mark detector
├── review.py               # Manual verification helpers & register matrix builders
├── export_data.py          # CSV & multi-sheet Excel exporter (retaining text Roll Numbers)
├── input_images/
│   └── attendance.jpg      # Sample photographed attendance register
├── output/
│   ├── preprocessed.jpg    # Contrast-enhanced grayscale register image
│   ├── students.csv        # Student list (S_No, Roll_No, Student_Name, Father_Name)
│   ├── attendance.csv      # Long-format attendance records with Confidence & Review_Flag
│   ├── attendance_matrix.csv # Register-style pivoted matrix (recorded sessions only)
│   ├── review_report.csv   # Uncertain/ambiguous cells flagged for human verification
│   └── attendance.xlsx     # Multi-sheet formatted Excel workbook
├── tests/
│   └── test_pipeline.py    # Automated Section 7 Testing Checklist verification suite
└── requirements.txt        # Python package dependencies
```

---

## 2. Attendance Rules Implemented

1. **Recorded Sessions with Dates Only**:
   - The system automatically detects which session columns under a date heading have recorded attendance marks (e.g., `2026-08-13_S1`, `2026-08-13_S2`, `2026-08-13_S3`) and ignores unrecorded future session columns (`S4–S6`).
2. **Binary `Present` / `Absent` Classification**:
   - **`P`** mark $\rightarrow$ **`Present`**
   - **Blank or any other value** $\rightarrow$ **`Absent`**
3. **Human-in-the-Loop Flagging**:
   - Ambiguous or non-standard marks (such as Row #15 `Deepanshu Kumar` in `2026-08-13_S3` which has two stray dots `..`) default to **`Absent`** per the rule above, and are automatically flagged in the **Manual Review Screen** and `review_report.csv` with a zoomed cell crop so staff can verify or override them.

---

## 3. How to Run the Application

### Launch the Streamlit Web App
```powershell
cd D:\ams
streamlit run app.py
```

### Run the Automated Verification Test Suite
```powershell
cd D:\ams
python tests/test_pipeline.py
```
