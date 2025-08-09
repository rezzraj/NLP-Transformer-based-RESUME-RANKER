import json
import pandas as pd
import re
from pandas import DataFrame

# Loading previously saved data into a pandas dataframe
with open("resume_texts.json", "r", encoding="utf-8") as f:
    resume_texts = json.load(f)
df=pd.DataFrame(list(resume_texts.items()), columns=["filename", "text"])


df['cleaned_text']=df['text'].str.lower()
print(df['cleaned_text'])

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
print(df['cleaned_text'])

