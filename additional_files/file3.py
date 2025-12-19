import pytesseract
from PIL import Image
import json
import google.generativeai as genai

# ----------------------------- #
# 1. OCR Extraction
# ----------------------------- #

def extract_text_from_image(image_path: str) -> str:
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text


# ----------------------------- #
# 2. LLM Table Extraction (Gemini)
# ----------------------------- #

def parse_table_with_llm(ocr_text: str, api_key: str):
    genai.configure(api_key=api_key)

    prompt = f"""
The following text comes from OCR of a very complex financial table
with multi-row headers, merged cells, and group columns.
Your job is to reconstruct this table into a clean hierarchical JSON.

### RULES ###
- Output MUST be valid JSON only.
- Handle multi-row headers and convert them into nested header groups.
- For each row, place data under the correct header group.
- If OCR missed a value, set it to null.
- DO NOT invent values.

### OUTPUT JSON FORMAT ###
{{
  "header_groups": [
      {{
         "group": "GroupName or null",
         "columns": ["col1", "col2", ...]
      }}
  ],
  "rows": [
      {{
         "row_label": "label text",
         "values": {{
             "GroupName": {{"col1": "...", "col2": "..."}},
             "AnotherGroup": {{...}}
         }}
      }}
  ]
}}

### OCR TEXT ###
{ocr_text}
"""

    model = genai.GenerativeModel("gemini-2.5-flash")   # better for structured tasks
    response = model.generate_content(prompt)

    try:
        return json.loads(response.text)
    except:
        print("MODEL RESPONSE:", response.text)
        raise ValueError("LLM did not return valid JSON")


# ----------------------------- #
# 3. Full Pipeline
# ----------------------------- #

def image_to_complex_json(image_path: str, api_key: str, output="table.json"):
    ocr_text = extract_text_from_image(image_path)
    json_data = parse_table_with_llm(ocr_text, api_key)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

    print(f"JSON saved as {output}")
    return json_data


# ----------------------------- #
# 4. Usage Example
# ----------------------------- #

if __name__ == "__main__":
    API_KEY = "AIzaSyCBprKqQPptyHGnjpuT_xuFs9Y0-9elIDE"
    image_path = "image/image.png"

    result = image_to_complex_json(image_path, API_KEY)
    print(json.dumps(result, indent=4))
