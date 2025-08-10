import json
import pandas as pd
import re
from pandas import DataFrame
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-mpnet-base-v2')

# Loading previously saved data into a pandas dataframe
with open("resume_texts.json", "r", encoding="utf-8") as f:
    resume_texts = json.load(f)
df=pd.DataFrame(list(resume_texts.items()), columns=["filename", "text"])

#performing text preprocessing
df['cleaned_text']=df['text'].str.lower()
def remove_special_chars(text):
    text=text.replace('\n',' ')
    text=re.sub(r'[^a-zA-Z0-9 .@\-/]','',text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    text = " ".join(text.split())
    return text.strip()
df['cleaned_text']=df['cleaned_text'].apply(lambda x: remove_special_chars(x))


cleaned_resumes='cleaned_resumes.json'

if os.path.exists(cleaned_resumes):
    df = pd.read_json(cleaned_resumes, orient='records', lines=True)  # ✅ pandas can read JSON Lines
    print("✅ Loaded cleaned resume saved data from JSON.")

else:
    df[['filename', 'cleaned_text']].to_json('cleaned_resumes.json', orient='records', lines=True)
    print('created a new json file')



