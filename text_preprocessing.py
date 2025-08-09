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


#embedding the resume data
texts=df['cleaned_text'].to_list()
npy_path='resume_embeddings.npy'
if os.path.exists(npy_path):
    resume_embedding= np.load(npy_path)
    print('loaded embeddings from npy file')
else:
    resume_embedding = model.encode(texts, batch_size=16, show_progress_bar=True)
    np.save(npy_path, resume_embedding)
    print('saved embeddings successfully')

job_description="""Seeking an AI/ML intern with strong Python skills, experience in machine learning frameworks like PyTorch or TensorFlow, 
knowledge of NLP techniques including transformers and embeddings, and ability to preprocess and analyze data. 
The candidate will assist in model development, training, fine-tuning, and evaluation, working closely with the team 
to build scalable AI solutions. Passion for learning and problem-solving is a must."""

job_dec_path='job_description_embedding.npy'
if os.path.exists(job_dec_path):
    job_embedding = np.load(job_dec_path)
    print('loaded job desc embeddings from npy file')
else:
    job_embedding = model.encode([job_description])
    np.save(job_dec_path, job_embedding)
    print('saved job desc embeddings successfully')
scores = cosine_similarity(job_embedding, resume_embedding)[0]
df['score'] = scores
df = df.sort_values(by='score', ascending=False)
print(df[['filename', 'score']])
