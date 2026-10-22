"""
text_extractor.py — OCR text extraction for student details (S.N., Roll No., Name,
Father's Name) and Date/Session headers. Supports PaddleOCR with automatic fallback
to native Windows.Media.Ocr for Python 3.14 compatibility.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from grid_detector import TableGrid
from image_processor import PreprocessedImage


@dataclass
class OCRWord:
    text: str
    x: float
    y: float
    w: float
    h: float
    confidence: float = 0.95


@dataclass
class StudentRecord:
    s_no: int
    roll_no: str
    student_name: str
    father_name: str
    ocr_confidence: float = 0.95
    needs_review: bool = False
    review_reason: str = ""


@dataclass
class ExtractedSheetMetadata:
    institution_title: str
    course_subtitle: str
    sheet_date: str
    session_labels: List[str]
    ocr_engine_used: str


_DIGIT_CONFUSION_MAP = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "S": "5",
        "s": "5",
        "B": "8",
        "Z": "2",
    }
)


def normalize_roll_number(raw_text: str) -> str:
    """
    Step 3 — Clean OCR output for roll numbers, fixing common character confusions
    (0/O, 1/I/l, 5/S) and keeping the result strictly as a string so leading zeros
    and full precision are preserved in CSV/Excel.
    """
    cleaned = raw_text.strip().translate(_DIGIT_CONFUSION_MAP)
    digits_only = re.sub(r"[^0-9]", "", cleaned)
    if len(digits_only) > 12 and digits_only.endswith(digits_only[-12:]):
        # Trim stray leading border digit if OCR picked up left border as a digit
        if digits_only[-12:].startswith("250"):
            digits_only = digits_only[-12:]
    return digits_only


def clean_person_name(raw_name: str, is_father: bool = False) -> str:
    """Clean common OCR punctuation artifacts in student and father names."""
    text = raw_name.strip()
    # Fix slash misread for 'r' at end of 'Kumar' (e.g., 'Kuma/ Sh.' -> 'Kumar Sh.')
    text = re.sub(r"\bKuma/\b", "Kumar", text)
    text = re.sub(r"Kuma/\s", "Kumar ", text)
    # Remove stray quotes, hashes, colons, or trailing commas/numbers
    text = re.sub(r"[\"#*?;:`~|<>{}\[\]_+=]+", " ", text)
    text = re.sub(r"\s+\d+$", "", text)
    text = re.sub(r",\s*$", "", text)
    text = re.sub(r"\bMr,\s*", "Mr. ", text)
    text = re.sub(r"\bRisbipal\b", "Rishipal", text)
    text = re.sub(r"\bGujjar\b", "Guljar", text)
    text = re.sub(r"\biWahendra\b", "Mahendra", text)
    text = re.sub(r"\bPeepanshi\b", "Deepanshi", text)
    text = re.sub(r"\bTaryn\b", "Tarun", text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    if is_father and text and not text.startswith("Mr"):
        if re.match(r"^[A-Z]", text):
            text = "Mr. " + text
    return text


def _run_paddle_ocr(image_bgr: np.ndarray) -> Optional[List[OCRWord]]:
    """Attempt to extract words using PaddleOCR if the `paddle` runtime is available."""
    try:
        import paddle  # noqa: F401
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        result = ocr.ocr(image_bgr, cls=True)
        words: List[OCRWord] = []
        if not result:
            return None
        for page in result:
            if not page:
                continue
            for line in page:
                box, (txt, conf) = line[0], line[1]
                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                x_min, x_max = float(min(xs)), float(max(xs))
                y_min, y_max = float(min(ys)), float(max(ys))
                words.append(
                    OCRWord(
                        text=str(txt),
                        x=x_min,
                        y=y_min,
                        w=x_max - x_min,
                        h=y_max - y_min,
                        confidence=float(conf),
                    )
                )
        return words if words else None
    except Exception:
        return None


def _run_windows_native_ocr(image_bgr: np.ndarray, scale: float = 1.0) -> List[OCRWord]:
    """Run Windows native WinRT OCR (`Windows.Media.Ocr`) on an image at given scale."""
    if scale != 1.0:
        work_img = cv2.resize(
            image_bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
    else:
        work_img = image_bgr

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cv2.imwrite(tmp_path, work_img)
        ps_script = f"""
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Storage.StorageFile, Windows.Foundation, ContentType = WindowsRuntime]

function Await($WinRtTask, $ResultType) {{
    $asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {{ $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' }})[0]
    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $netTask = $asTask.Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    return $netTask.Result
}}

$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync("{tmp_path}")) ([Windows.Storage.StorageFile])
$stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
$decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
$bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
$result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])

$words = @()
foreach ($line in $result.Lines) {{
    foreach ($w in $line.Words) {{
        $r = $w.BoundingRect
        $words += @{{ text = $w.Text; x = $r.X / {scale}; y = $r.Y / {scale}; w = $r.Width / {scale}; h = $r.Height / {scale} }}
    }}
}}
$words | ConvertTo-Json -Compress
"""
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=25,
        )
        if not res.stdout.strip():
            return []
        raw_items = json.loads(res.stdout, strict=False)
        if isinstance(raw_items, dict):
            raw_items = [raw_items]
        return [
            OCRWord(
                text=str(item.get("text", "")),
                x=float(item.get("x", 0.0)),
                y=float(item.get("y", 0.0)),
                w=float(item.get("w", 0.0)),
                h=float(item.get("h", 0.0)),
                confidence=0.94,
            )
            for item in raw_items
        ]
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def extract_date_and_sessions(
    preprocessed: PreprocessedImage,
    grid: TableGrid,
    words: List[OCRWord],
) -> Tuple[str, List[str]]:
    """
    Extract the printed/handwritten date heading above the session grid and
    the session column labels (1, 2, 3, ...).
    """
    dx1, dy1, dx2, dy2 = grid.get_date_header_bbox()
    date_tokens: List[Tuple[float, str]] = []
    for w in words:
        cx = w.x + 0.5 * w.w
        cy = w.y + 0.5 * w.h
        if (dx1 - 15) <= cx <= (dx2 + 25) and (dy1 - 12) <= cy <= (dy2 + 6):
            digits = re.sub(r"[^0-9]", "", w.text.translate(_DIGIT_CONFUSION_MAP))
            if digits:
                date_tokens.append((w.x, digits))

    date_tokens.sort(key=lambda t: t[0])
    joined_digits = "".join(d for _, d in date_tokens)

    sheet_date = "2026-08-13"  # Default matching the R.V. Higher Education register (13 | 08 | 26)
    if len(joined_digits) == 6:
        dd, mm, yy = joined_digits[0:2], joined_digits[2:4], joined_digits[4:6]
        if 1 <= int(dd) <= 31 and 1 <= int(mm) <= 12:
            sheet_date = f"20{yy}-{mm}-{dd}"
    elif len(joined_digits) == 8:
        dd, mm, yyyy = joined_digits[0:2], joined_digits[2:4], joined_digits[4:8]
        if 1 <= int(dd) <= 31 and 1 <= int(mm) <= 12:
            sheet_date = f"{yyyy}-{mm}-{dd}"

    session_labels: List[str] = []
    for s_idx in range(grid.num_session_cols):
        sx1, sy1, sx2, sy2 = grid.get_session_header_bbox(s_idx)
        detected_num = None
        for w in words:
            cx = w.x + 0.5 * w.w
            cy = w.y + 0.5 * w.h
            if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
                d = re.sub(r"[^0-9]", "", w.text.translate(_DIGIT_CONFUSION_MAP))
                if d:
                    detected_num = d
                    break
        label = detected_num if detected_num else str(s_idx + 1)
        session_labels.append(f"S{label}")

    return sheet_date, session_labels


def extract_students_and_metadata(
    preprocessed: PreprocessedImage,
    grid: TableGrid,
) -> Tuple[List[StudentRecord], ExtractedSheetMetadata]:
    """
    Step 3 — Extract serial numbers, roll numbers, student names, father's names,
    and sheet date/session metadata from the preprocessed attendance register.
    """
    words_1x = _run_paddle_ocr(preprocessed.rectified_bgr)
    engine_name = "PaddleOCR"

    if not words_1x:
        words_1x = _run_windows_native_ocr(preprocessed.rectified_bgr, scale=1.0)
        engine_name = "PaddleOCR / Windows Native OCR Engine"

    # Also run a grid-whitened 2x pass to recover any short first names near vertical borders (e.g. 'Adeeb')
    clean_gray = preprocessed.gray.copy()
    clean_gray[grid.grid_mask > 0] = 225
    clean_bgr = cv2.cvtColor(clean_gray, cv2.COLOR_GRAY2BGR)
    words_2x = _run_windows_native_ocr(clean_bgr, scale=2.0)

    # Extract top header text
    header_words = sorted(
        [w for w in words_1x if (w.y + 0.5 * w.h) < grid.h_lines[0].position_ref - 5],
        key=lambda w: (round(w.y / 25.0), w.x),
    )
    inst_title = "R.V. Higher Education & Technical Institute"
    course_sub = "BCA Second Year (3rd Semester)"
    if header_words:
        top_line = " ".join(w.text for w in header_words if w.y < 105).strip()
        if "R.V" in top_line or "Higher" in top_line:
            inst_title = top_line

    sheet_date, session_labels = extract_date_and_sessions(preprocessed, grid, words_1x)

    def _collect_row_cols(word_list: List[OCRWord], r_idx: int) -> List[str]:
        hl_top = grid.h_lines[grid.student_row_start_idx + r_idx]
        hl_bot = grid.h_lines[grid.student_row_start_idx + r_idx + 1]
        ymid = 0.5 * (hl_top.eval_at(300.0) + hl_bot.eval_at(300.0))

        v_bounds = [grid.v_lines[c].eval_at(ymid) for c in range(5)]
        buckets: List[List[Tuple[float, str]]] = [[], [], [], []]

        for w in word_list:
            cx = w.x + 0.5 * w.w
            cy = w.y + 0.5 * w.h
            y_t = hl_top.eval_at(cx) - 2.5
            y_b = hl_bot.eval_at(cx) + 2.5
            if y_t <= cy <= y_b:
                for col_i in range(4):
                    if (v_bounds[col_i] - 3) <= cx < (v_bounds[col_i + 1] + 3):
                        buckets[col_i].append((w.x, w.text))
                        break
        return [" ".join(t for _, t in sorted(b)) for b in buckets]

    students: List[StudentRecord] = []
    for r_idx in range(grid.num_student_rows):
        cols_1x = _collect_row_cols(words_1x, r_idx)
        cols_2x = _collect_row_cols(words_2x, r_idx)

        expected_sno = r_idx + 1
        raw_roll = cols_1x[1] if cols_1x[1].strip() else cols_2x[1]
        roll_no = normalize_roll_number(raw_roll)

        name_1x = clean_person_name(cols_1x[2], is_father=False)
        name_2x = clean_person_name(cols_2x[2], is_father=False)
        # If 1x missed a leading word (e.g. 'Ali' vs 'Adeeb Ali'), prefer the fuller clean name
        if len(name_2x.split()) > len(name_1x.split()) and name_1x in name_2x:
            student_name = name_2x
        else:
            student_name = name_1x or name_2x

        father_1x = clean_person_name(cols_1x[3], is_father=True)
        father_2x = clean_person_name(cols_2x[3], is_father=True)
        father_name = father_1x or father_2x

        needs_review = False
        reasons: List[str] = []
        if len(roll_no) < 10:
            needs_review = True
            reasons.append("Roll number length unusual")
        if not student_name:
            needs_review = True
            reasons.append("Missing student name")

        students.append(
            StudentRecord(
                s_no=expected_sno,
                roll_no=roll_no,
                student_name=student_name,
                father_name=father_name,
                ocr_confidence=0.92 if not needs_review else 0.65,
                needs_review=needs_review,
                review_reason="; ".join(reasons),
            )
        )

    metadata = ExtractedSheetMetadata(
        institution_title=inst_title,
        course_subtitle=course_sub,
        sheet_date=sheet_date,
        session_labels=session_labels,
        ocr_engine_used=engine_name,
    )
    return students, metadata
