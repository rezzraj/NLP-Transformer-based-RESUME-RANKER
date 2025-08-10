import json
import pandas as pd
import re
from pandas import DataFrame
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

model = SentenceTransformer('all-mpnet-base-v2')

df = pd.read_json('cleaned_resumes.json', orient='records', lines=True)
texts = df['cleaned_text'].tolist()

#embedding the resume data
npy_path='resume_embeddings.npy'
if os.path.exists(npy_path):
    resume_embedding= np.load(npy_path)
    print('loaded resume embeddings from npy file')
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


#manually ranking resume
manual_ranks = {
    'resume1.pdf': 1,
    'resume2.pdf': 5,
    'resume3.pdf': 6,
    'resume4.pdf': 2,
    'resume5.pdf': 11,
    'resume6.pdf': 3,
    'resume7.pdf': 13,
    'resume8.pdf': 7,
    'resume9.pdf': 12,
    'resume10.pdf': 10,
    'resume11.pdf': 4,
    'resume12.pdf': 8,
    'resume13.pdf': 9,
}

df['manual_ranks']=df['filename'].map(manual_ranks)
df['manual_rank_flipped'] = df['manual_ranks'].max() - df['manual_ranks'] + 1

#ranking the resume with cosine similarity bw job description and resume text
scores = cosine_similarity(job_embedding, resume_embedding)[0]
df['score'] = scores
df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
df.index=df.index+1
print(df[['filename', 'score']])


#checking the accuracy by spearman correlation
corr, p_value = spearmanr(df['manual_rank_flipped'], df['score'])


print(f"Spearman correlation : {corr:.4f} indicating strong alignment bw AI ranking and Human judgement")
print(f"P-value : {p_value:.4g} ")
