import os
import pandas as pd
import numpy as np
import string
import nltk
import scipy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

# Loading Bert transformer pre-trained model.
from transformers import AutoTokenizer, AutoModel, AutoConfig,BertTokenizer,BertModel, AutoModelForTokenClassification
import torch
import torch.nn.functional as F

# Specify the new embedding size you want
new_embedding_size = 768

# Importing FastAPI
from fastapi import Body, FastAPI
app = FastAPI()



import subprocess

def git(*args):
    return subprocess.check_call(['git'] + list(args))

# examples
try:
    git("clone", "https://huggingface.co/datasets/mozay22/medical_code_mapping")
except:
    pass

icd_path = os.path.join(os.getcwd(),"medical_code_mapping/ICD_10_CODE_2023.csv")
icd_codes = pd.read_csv(icd_path)
icd_codes["LONG_DESCRIPTION"].fillna("unknown", inplace = True);icd_codes["SHORT_DESCRIPTION"].fillna("unknown", inplace = True);
# print(icd_codes.head(5))

cpt_path = os.path.join(os.getcwd(),"medical_code_mapping/cpt_codes_consolidated.csv")
cpt_codes = pd.read_csv(cpt_path)
# cpt_codes.head(5)


# Importing response functions
from utilities.function import *

long_descrption_embedding_load = torch.load(os.path.join(os.getcwd(),"medical_code_mapping/clinical_diag_description_pytorch_large.pt"))
cpt_descrption_embedding_load = torch.load(os.path.join(os.getcwd(),"medical_code_mapping/clinical_cpt_description_pytorch_large.pt"))

@app.post("/code_mapping/")
async def create_book(_body_= Body()):
    dict_= diag_and_cpt_tagging(text = str(_body_['BODY']),
                                diag_embedding = long_descrption_embedding_load, 
                                cpt_embedding=cpt_descrption_embedding_load,
                                diag_df = icd_codes,
                                cpt_df = cpt_codes,
                                diag_key = _body_['DIAG_KEY'],
                                cpt_key= _body_['CPT_KEY'],
                                _threshold_ = _body_['THRESHOLD'] , # default 0.2
                                diag_rank = _body_['DIAG_TOP_RANK'], #DEFAULT 2
                                cpt_rank = _body_['CPT_TOP_RANK']) # DEFAULT 2
    return dict_
