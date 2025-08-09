import json
import pandas as pd
import re
from pandas import DataFrame
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('all-mpnet-base-v2')

df = pd.read_json('cleaned_resumes.json', orient='records', lines=True)
texts = df['cleaned_text'].tolist()

#embedding the resume data
npy_path='resume_embeddings.npy'
if os.path.exists(npy_path):
    resume_embedding= np.load(npy_path)
    print('loaded embeddings from npy file')
else:
    resume_embedding = model.encode(texts, batch_size=16, show_progress_bar=True)
    np.save(npy_path, resume_embedding)
    print('saved embeddings successfully')

#opening job description text file from the folder

with open('job_description.txt', 'r', encoding='utf-8') as f:
    job_description = f.read()

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
