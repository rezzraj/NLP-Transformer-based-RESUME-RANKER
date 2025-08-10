import os
import re
import json
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Paths
folder_path = r"resume pdfs(data)"
json_path = "resume_texts.json"

# Function for natural sorting (so resume2 < resume10)
def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

resume_texts = {}

if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        resume_texts = json.load(f)
    print("✅ Loaded saved data from JSON.")
else:
    # Sort files naturally
    pdf_files = sorted(
        [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")],
        key=natural_key
    )

    for file in pdf_files:
        pdf_path = os.path.join(folder_path, file)

        # Convert PDF pages to images
        pages = convert_from_path(pdf_path)

        # Extract text per page
        text = ""
        for page in pages:
            text += pytesseract.image_to_string(page) + "\n"

        # Store in dict with exact filename mapping
        resume_texts[file] = text.strip()

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resume_texts, f, indent=4, ensure_ascii=False)
    print("✅ OCR complete & data saved to JSON.")

# Final output check
print(f"Total resumes processed: {len(resume_texts)}")
print(list(resume_texts.keys())[:5])  # Show first few filenames
