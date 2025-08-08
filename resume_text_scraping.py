import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import json

# If on Windows, set tesseract path
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

folder_path = r"C:\Users\akshi\Downloads\converted_pdfs"
json_path = "resume_texts.json"

resume_texts = {}

# If JSON already exists, load it
if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        resume_texts = json.load(f)
    print("✅ Loaded saved data from JSON.")
else:
    # Process PDFs if JSON not found
    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)

            # Convert PDF to images
            pages = convert_from_path(pdf_path)

            text = ""
            for page in pages:
                text += pytesseract.image_to_string(page) + "\n"

            # Store in dict
            resume_texts[file] = text.strip()

    # Save to JSON for later use
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resume_texts, f, indent=4, ensure_ascii=False)
    print("💾 OCR complete & data saved to JSON.")

# Now resume_texts has your data
print(resume_texts)
