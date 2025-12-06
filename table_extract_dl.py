#!/usr/bin/env python3
"""
Production-style dynamic table extraction using:
  - layoutparser (PubLayNet) for table detection (deep learning)
  - robust cell segmentation via line detection + intersections on the detected table region
  - pytesseract OCR per cell
  - build Option B JSON: list of row-dicts using detected header row as keys
  - fallback segmentation if line-based method fails

Outputs:
  - JSON file (list of objects)
  - CSV file

Usage:
  python table_extract_dl.py --image /path/to/image.png --outdir ./out
"""

import os
import argparse
import json
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
import pytesseract
import pandas as pd

# Try to import layoutparser; if not available, instruct to install.
try:
    import layoutparser as lp
except Exception as e:
    lp = None
    print("layoutparser is not installed or failed to import.")
    print("Install with: pip install layoutparser[ocr,paddledetection]")
    print("Then re-run the script.")
    # We'll still provide a fallback path that uses OpenCV-only detection (but deep detection is preferred).

# ---------- Helper utilities ----------

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def pil_from_cv2(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def ocr_image_to_text(img_gray_or_pil) -> str:
    """OCR single ROI (PIL Image or grayscale numpy). Return stripped text."""
    if isinstance(img_gray_or_pil, np.ndarray):
        pil = Image.fromarray(img_gray_or_pil)
    else:
        pil = img_gray_or_pil
    # Tesseract config - assume small text, take single block
    config = "--psm 6"  # assume a uniform block of text
    text = pytesseract.image_to_string(pil, config=config)
    return text.strip().replace("\n", " ").strip()

# ---------- Cell segmentation inside a table crop ----------
def detect_cells_by_lines(table_img_gray: np.ndarray) -> List[Tuple[int,int,int,int]]:
    """
    Detect cell bounding boxes by detecting horizontal and vertical lines,
    finding their intersections and producing a grid of bounding boxes.
    Returns list of (x,y,w,h) for each detected cell.
    """
    # Use adaptive threshold and invert for lines detection
    img = table_img_gray.copy()
    if img.dtype != np.uint8:
        img = (img*255).astype('uint8')
    # Binarize
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = 255 - bw  # invert: lines become white

    h, w = bw.shape

    # kernels: tune sizes relative to image
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w//30), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h//30)))

    horiz = cv2.erode(bw, horizontal_kernel, iterations=1)
    horiz = cv2.dilate(horiz, horizontal_kernel, iterations=1)

    vert = cv2.erode(bw, vertical_kernel, iterations=1)
    vert = cv2.dilate(vert, vertical_kernel, iterations=1)

    # Combine lines
    grid = cv2.add(horiz, vert)

    # Find intersections (bitwise_and)
    intersections = cv2.bitwise_and(horiz, vert)

    # Find contours of grid boxes
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x,y,wc,hc = cv2.boundingRect(cnt)
        # Filter sizes
        if wc < max(20, w*0.02) or hc < max(10, h*0.02):
            continue
        boxes.append((x,y,wc,hc))

    # If too few boxes, return empty to indicate failure
    if len(boxes) < 6:
        return []

    # Now try to build a grid: cluster boxes by row (y) and col (x)
    boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0]))
    # group rows by y proximity
    rows = []
    cur_row = [boxes_sorted[0]]
    for b in boxes_sorted[1:]:
        if abs(b[1] - cur_row[-1][1]) < max(10, int(cur_row[-1][3]*0.5)):
            cur_row.append(b)
        else:
            rows.append(sorted(cur_row, key=lambda bb: bb[0]))
            cur_row = [b]
    if cur_row:
        rows.append(sorted(cur_row, key=lambda bb: bb[0]))

    # flatten rows to cells list (sorted top-left to bottom-right)
    cells = []
    for r in rows:
        cells.extend(r)
    return cells

def fallback_split_grid(table_img_gray: np.ndarray, expected_cols:int=10) -> List[Tuple[int,int,int,int]]:
    """
    Fallback: split the table image into a grid by estimating rows from horizontal projections and columns by equal width split (or use vertical projection).
    This is used when line-based segmentation fails.
    """
    img = table_img_gray.copy()
    h, w = img.shape
    # find horizontal splits by projection
    thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21,10)
    # compute horizontal projection (count of dark pixels per row)
    proj = (thresh==0).sum(axis=1)
    # detect peaks (rows with content)
    import scipy.signal as ss
    peaks, _ = ss.find_peaks(proj, distance=max(5, h//50), height=np.max(proj)*0.1)
    # create row boundaries from peaks; if too few, use fixed row height heuristic
    if len(peaks) < 3:
        # estimate 12 rows by heuristic (makes dynamic)
        est_rows = max(3, h//40)
        row_h = max(10, h // est_rows)
        row_bounds = [(i*row_h, min(h, (i+1)*row_h)) for i in range(est_rows)]
    else:
        # build row bounds around peaks
        row_bounds = []
        # turn peaks into ranges by finding midpoints between adjacent peaks
        boundaries = [0]
        for i in range(len(peaks)-1):
            boundaries.append((peaks[i]+peaks[i+1])//2)
        boundaries.append(h)
        for i in range(len(boundaries)-1):
            row_bounds.append((boundaries[i], boundaries[i+1]))

    # columns: try vertical projection; else equal width split
    vproj = (thresh==0).sum(axis=0)
    # find significant gaps to split columns
    cols_idxs = []
    gap_threshold = np.max(vproj)*0.05
    current_start = 0
    # naive: create expected_cols equal width splits if projection is noisy
    if np.max(vproj) < 5:
        # fallback equal splits
        cols = expected_cols
        col_w = max(10, w // cols)
        cols_idxs = [(i*col_w, min(w, (i+1)*col_w)) for i in range(cols)]
    else:
        # try to detect separators by scanning for long vertical white area
        white_cols = np.where(vproj < gap_threshold)[0]
        if len(white_cols) < 2:
            # fallback equal splits
            cols = expected_cols
            col_w = max(10, w // cols)
            cols_idxs = [(i*col_w, min(w, (i+1)*col_w)) for i in range(cols)]
        else:
            # make splits at large white gaps
            splits = []
            runs = np.split(white_cols, np.where(np.diff(white_cols) != 1)[0]+1)
            for r in runs:
                if r.size > max(2, h//100):
                    # use midpoint
                    splits.append(r.mean())
            # include 0 and w
            boundaries = [0] + [int(x) for x in splits] + [w]
            cols_idxs = []
            for i in range(len(boundaries)-1):
                cols_idxs.append((boundaries[i], boundaries[i+1]))

    # compose boxes from rows x cols
    boxes = []
    for rb in row_bounds:
        ry0, ry1 = rb
        for cb in cols_idxs:
            cx0, cx1 = cb
            ww = cx1 - cx0
            hh = ry1 - ry0
            if ww < 10 or hh < 10:
                continue
            boxes.append((cx0, ry0, ww, hh))
    # sort boxes
    boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes_sorted

# ---------- Main pipeline ----------
def extract_tables_from_image(image_path: str, out_dir: str, use_layoutparser=True):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    detected_tables = []

    # Step 1: detect table regions using layoutparser deep detector (PubLayNet)
    if use_layoutparser and lp is not None:
        try:
            # model from layoutparser model zoo (PubLayNet)
            model = lp.PaddleDetectionLayoutModel('lp://PubLayNet/ppyolov2_r50vd_dcn_365e/config', threshold=0.5)
            layout = model.detect(img_pil)
            # pick blocks of type 'Table' (layoutparser standard)
            table_blocks = [b for b in layout if b.type.lower() == 'table']
            for tb in table_blocks:
                x1, y1, x2, y2 = map(int, tb.block.coordinates)
                # crop
                crop = img_bgr[y1:y2, x1:x2]
                detected_tables.append(((x1,y1,x2,y2), crop))
        except Exception as e:
            print("layoutparser detection failed:", e)
            detected_tables = []

    # If no tables found by deep model, try simple heuristic: detect largest rectangle-like region with many lines
    if not detected_tables:
        print("No table detected by deep model — falling back to heuristic whole-image table detection.")
        # You could try scanning for the largest connected component in morphological grid
        # For simplicity, treat the entire image as one table region
        detected_tables.append(((0,0,img_bgr.shape[1], img_bgr.shape[0]), img_bgr))

    results_all_tables = []

    for idx, (bbox, table_crop_bgr) in enumerate(detected_tables):
        print(f"Processing table #{idx+1} at bbox {bbox}")
        table_gray = cv2.cvtColor(table_crop_bgr, cv2.COLOR_BGR2GRAY)

        # First: try robust line-based cell detection
        cells = detect_cells_by_lines(table_gray)

        # If line-based detection failed or detected too few cells, fallback to hybrid/fallback splitting
        if len(cells) < 6:
            print("Line-based cell detection returned too few cells; using fallback grid split.")
            # attempt detection with expected columns estimated by header detection
            cells = fallback_split_grid(table_gray, expected_cols=10)

        # If still no cells, skip
        if not cells:
            print("No cells detected for this table. Skipping.")
            continue

        # Convert global coordinates: cells are relative to crop, we don't need global later but keep if needed
        # Now cluster cells into rows by y coordinate
        cells_sorted = sorted(cells, key=lambda b: (b[1], b[0]))
        # group into rows
        rows = []
        cur_row = [cells_sorted[0]]
        for b in cells_sorted[1:]:
            if abs(b[1] - cur_row[-1][1]) < max(10, int(cur_row[-1][3]*0.6)):
                cur_row.append(b)
            else:
                rows.append(sorted(cur_row, key=lambda bb: bb[0]))
                cur_row = [b]
        if cur_row:
            rows.append(sorted(cur_row, key=lambda bb: bb[0]))

        # Build 2D list of cell texts by OCR
        table_text_grid = []
        for r in rows:
            row_texts = []
            for (x,y,wc,hc) in r:
                # extract ROI with small padding
                pad = 3
                x0 = max(0, x-pad); y0 = max(0, y-pad)
                x1 = min(table_gray.shape[1], x+wc+pad); y1 = min(table_gray.shape[0], y+hc+pad)
                roi = table_gray[y0:y1, x0:x1]
                # optional: deskew small ROI (skipped for speed)
                # binarize + OCR
                _, roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                txt = ocr_image_to_text(roi_bin)
                row_texts.append(txt)
            table_text_grid.append(row_texts)

        # Normalize each row to the same column count (pad with "")
        max_cols = max(len(r) for r in table_text_grid)
        for r in table_text_grid:
            if len(r) < max_cols:
                r.extend([""] * (max_cols - len(r)))

        # Heuristic: find header row (first row that contains words like 'Published', 'Price', 'Strength' or row with many non-empty cells)
        header_row_idx = 0
        for i, r in enumerate(table_text_grid[:4]):  # look at first few rows
            joined = " ".join(r).lower()
            if any(k in joined for k in ["published", "price", "strength", "oct", "published/"]):
                header_row_idx = i
                break
            # else choose row with most non-empty fields
            if i == 0 and sum(1 for c in r if c.strip()) >= max(2, max_cols//2):
                header_row_idx = 0

        header = [c if c.strip() else f"col{j+1}" for j,c in enumerate(table_text_grid[header_row_idx])]

        # Build output objects: each subsequent row becomes an object keyed by header
        objects = []
        for r_idx, r in enumerate(table_text_grid[header_row_idx+1:], start=header_row_idx+1):
            # skip empty rows heuristically
            if all(not (c.strip()) for c in r):
                continue
            obj = {}
            for ci, key in enumerate(header):
                val = r[ci] if ci < len(r) else ""
                obj[key] = val
            objects.append(obj)

        results_all_tables.append({
            "bbox": bbox,
            "header": header,
            "rows": table_text_grid,
            "objects": objects
        })

    # Merge results (if multiple tables, concatenate rows)
    merged_objects = []
    for t in results_all_tables:
        merged_objects.extend(t["objects"])

    # If JSON-empty, still write what we have (maybe header only)
    out_json_path = os.path.join(out_dir, "extracted_table_optionB.json")
    out_csv_path  = os.path.join(out_dir, "extracted_table_optionB.csv")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(merged_objects, f, indent=2, ensure_ascii=False)

    # Save CSV via pandas (if objects empty, create empty DF)
    if merged_objects:
        df = pd.DataFrame(merged_objects)
    else:
        df = pd.DataFrame()
    df.to_csv(out_csv_path, index=False)

    return {
        "json": out_json_path,
        "csv": out_csv_path,
        "tables": results_all_tables
    }


# ---------- Command line ----------
def main():
    parser = argparse.ArgumentParser(description="Dynamic table extraction (DL + robust segmentation + OCR) -> Option B JSON")
    parser.add_argument("--image", required=True, help="Path to table image")
    parser.add_argument("--outdir", default="./out", help="Directory for outputs")
    parser.add_argument("--no-layoutparser", action="store_true", help="Do not use layoutparser deep detector (force fallback)")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    use_lp = not args.no_layoutparser
    result = extract_tables_from_image(args.image, args.outdir, use_layoutparser=use_lp)

    print("Done.")
    print("JSON:", result["json"])
    print("CSV:", result["csv"])
    # print a short preview
    with open(result["json"], "r", encoding="utf-8") as f:
        data = json.load(f)
    print("Extracted rows (preview):", data[:5])

if __name__ == "__main__":
    main()
