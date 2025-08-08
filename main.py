import json

# Load previously saved data
with open("resume_texts.json", "r", encoding="utf-8") as f:
    resume_texts = json.load(f)

# Now you can use resume_texts just like before
print(resume_texts.keys())  # prints all filenames
print(resume_texts["resume1.pdf"])  # prints text of a specific file
