from PIL import Image
import pytesseract
import cv2
import json
import pandas as pd

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def image_to_json_and_csv(image_path, json_path, csv_path):
    processed = preprocess_image(image_path)
    pil_img = Image.fromarray(processed)

    text = pytesseract.image_to_string(pil_img, config="--psm 6")
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    header = lines[0].split()
    rows = []

    for line in lines[1:]:
        cells = line.split()
        if len(cells) < len(header):
            cells += [""] * (len(header) - len(cells))
        elif len(cells) > len(header):
            cells = cells[:len(header)]
        rows.append(dict(zip(header, cells)))

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    # JSON → CSV
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

# Example usage
image_to_json_and_csv("image/image.png", "table.json", "table.csv")
