import pytesseract
from PIL import Image
import json
import csv
import google.generativeai as genai
import re


# ---------------------------------------------------------
# 1. OCR Extraction from Image
# ---------------------------------------------------------

# def extract_text_from_image(image_path: str) -> str:
#     img = Image.open(image_path)
#     text = pytesseract.image_to_string(img)
#     print(text)
#     return text


# ---------------------------------------------------------
# 2. Convert OCR Output to JSON Using Gemini
# ---------------------------------------------------------

def convert_table_text_to_json(table_text: str, api_key: str):
    genai.configure(api_key=api_key)

    prompt = f"""
    Convert the following image into a clean JSON table.

    Requirements:
    - Detect columns accurately.
    - Detect all rows.
    - Do not include any calculations. Keep it as it is. Every expressions should be intact.
    - Do not include explanations.
    - Output ONLY valid JSON.
    - JSON format:
        {{
            "columns": [...],
            "rows": [
                {{ "col1": "...", "col2": "...", ... }},
                ...
            ]
        }}

    OCR text:
    {table_text}
    """

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    try:
        return json.loads(response.text)
    except Exception as e:
        print("LLM returned invalid JSON. Raw output saved instead.")
        return response.text


# ---------------------------------------------------------
# 3. Save JSON to file: table.json
# ---------------------------------------------------------

def save_json_to_file(data, filename="table.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"JSON saved → {filename}")


# ---------------------------------------------------------
# 4. Convert JSON → CSV (table.csv)
# ---------------------------------------------------------
def force_json_parse(text: str):
    """
    Extracts the JSON portion from LLM output even if the output
    has comments, markdown, explanations, or stray characters.
    """
    try:
        # Try direct parse
        return json.loads(text)
    except:
        pass

    # Try extracting JSON block with regex
    json_match = re.search(r"{[\s\S]*}", text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass

    raise ValueError("LLM did not return valid JSON:\n" + text)


def json_to_csv(json_file: str, csv_file: str):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, str):
        data = force_json_parse(data)

    columns = data.get("columns", [])
    rows = data.get("rows", [])

    if not columns or not rows:
        raise ValueError("JSON missing 'columns' or 'rows'")

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            filtered = {col: row.get(col, "") for col in columns}
            writer.writerow(filtered)

    print(f"CSV saved → {csv_file}")


# ---------------------------------------------------------
# 5. Master Pipeline Wrapper
# ---------------------------------------------------------

def image_to_csv_pipeline(image_path: str, api_key: str):
    print("\n🔍 Extracting text from image...")
    # ocr_text = extract_text_from_image(image_path)

    print("🤖 Sending to Gemini for JSON conversion...")
    json_data = convert_table_text_to_json(image_path, api_key)

    print("💾 Saving JSON...")
    save_json_to_file(json_data, "table.json")

    print("📄 Converting JSON → CSV...")
    json_to_csv("table.json", "table.csv")

    print("\n✅ Pipeline completed successfully!")
    print("➡ Output files: table.json, table.csv")


# ---------------------------------------------------------
# 6. Example Usage
# ---------------------------------------------------------

if __name__ == "__main__":
    API_KEY = "AIzaSyCBprKqQPptyHGnjpuT_xuFs9Y0-9elIDE"
    image_path = "image/image.png"
    

    image_to_csv_pipeline(image_path, API_KEY)

    

