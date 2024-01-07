import re
import os
import pandas as pd
import numpy as np
import string
import nltk
import scipy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Loading Bert transformer pre-trained model.
from transformers import AutoTokenizer, AutoModel, AutoConfig,BertTokenizer,BertModel, AutoModelForTokenClassification
import torch
import torch.nn.functional as F
new_embedding_size = 768


# # Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


def extract_uppercase_key_phrases(text):
    # Use a regular expression to find uppercase key phrases ending with ":"
    key_phrases = re.findall(r'\b[A-Z][A-Z\s]+\b:', text)

    # Remove the trailing colon from each key phrase
    key_phrases = [phrase.rstrip(':') for phrase in key_phrases]

    return key_phrases


def extract_text_after(input_word, corpus, stop_words=[]):
    """
    Extract text after the input word in the corpus until a stop word is encountered.

    Parameters:
    - input_word: The word or phrase to start extraction from.
    - corpus: The text corpus to extract from.
    - stop_words: A list of words or phrases to stop extraction when encountered.

    Returns:
    - Extracted text.
    """
    # Find the position of the input word in the corpus
    #  input word index
    start_index = corpus.find(input_word)

    stop_index_list = []
    for words in stop_words:
        stop_index_list.append(corpus.find(words))

    if start_index == max(stop_index_list):
      extracted_text = corpus[start_index + len(input_word):]
      return extracted_text
    else:
      # Find the position of the first stop word in the corpus
      stop_index = min([_index_ for _index_ in stop_index_list if _index_ > start_index])
      extracted_text = corpus[start_index + len(input_word):stop_index]
      return extracted_text



# Split sentence in single words.
def split_sentences_into_words(sentences):
    """
    Split a list of sentences into a flat list of words.

    Parameters:
    - sentences (list): A list of sentences.

    Returns:
    - list: A flat list containing all the words from the input sentences.
    """
    words_list = [word for sentence in sentences for word in sentence.split()]
    return words_list


def remove_duplicates(input_list):
    """
    Remove duplicates from a list while maintaining the original order.

    Parameters:
    - input_list (list): The input list with possible duplicates.

    Returns:
    - list: A new list with duplicates removed and original order maintained.
    """
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result



def remove_stopwords_in_wordlist(wordlist):
    """
    Remove stopwords from a word list.

    Parameters:
    - corpus (list): List of words.
    Returns:
    - list: List of word excluding stopwords.
    """
    stop_words = set(stopwords.words('english'))
    cleaned_corpus = [word.upper() for word in wordlist if word.lower() not in stop_words]
    return cleaned_corpus


# Remove cutom stopswords
def remove_custom_stopwords(corpus, custom_list):
    """
    Remove custom stopwords from a corpus.

    Parameters:
    - corpus (list): List of sentences or documents in the corpus.

    Returns:
    - list: List of sentences or documents with stopwords removed.
    """
    stop_words = set(custom_list)
    word_tokens = word_tokenize(corpus)
    filtered_words = [word for word in word_tokens if word.upper() not in stop_words]
    filtered_words_corpus = " ".join(filtered_words)

    return filtered_words_corpus



def replace_punctuation_with_space(text):
    """
    Replace punctuation in a text with spaces.

    Parameters:
    - text (str): The input text containing punctuation.

    Returns:
    - str: The text with punctuation replaced by spaces.
    """
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text_without_punctuation = text.translate(translator)
    return text_without_punctuation



def replace_punctuation_with_space_except_period(text):
    """
    Replace punctuation in a text with spaces, excluding the period.

    Parameters:
    - text (str): The input text containing punctuation.

    Returns:
    - str: The text with punctuation (except period) replaced by spaces.
    """
    # Create a translation table, excluding the period
    punctuation_except_period = string.punctuation.replace('.', '')
    translator = str.maketrans(punctuation_except_period, ' ' * len(punctuation_except_period))

    # Replace punctuation with spaces, except period
    text_without_punctuation = text.translate(translator)
    return text_without_punctuation


def corpus_cleaning(x, remove_emojis=True, remove_stop_words=False, stop_words = stopwords.words("english")):
    """Apply function to a clean a corpus"""
    x = x.lower().strip()
    stop_words = [x.upper() for x in stop_words]
    # romove urls
    url = re.compile(r'https?://\S+|www\.\S+')
    x = url.sub(r'',x)
    # remove html tags
    html = re.compile(r'<.*?>')
    x = html.sub(r'',x)
    # remove punctuation
    x = replace_punctuation_with_space_except_period(x)
    if remove_emojis:
        x = x.encode('ascii', 'ignore').decode('utf8').strip()
    if remove_stop_words:
        x = ' '.join([word for word in x.split(' ') if word not in stop_words])
    return x

#Mean Pooling - Take average of all tokens
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


#Encode text
def encode(texts):
    # Tokenize sentences
    encoded_input = tokenizer(texts,  padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)

    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


#################################### Drop short strings ####################################
def drop_short_strings(input_list, min_length=10):
    """
    Remove strings from a list if their size is less than a specified minimum length.

    Parameters:
    - input_list (list): The input list of strings.
    - min_length (int): The minimum length for a string to be retained (default is 10).

    Returns:
    - list: The list with strings removed if their size is less than the specified minimum length.
    """
    filtered_list = [string for string in input_list if len(string) >= min_length]
    return filtered_list


# ################################################ Text Embedding with loop ################################################
def tokenize_embedding_loop(iter_list,loop_split=10000, embbedding_size = new_embedding_size):
    iter_list_len = len(iter_list)
    tensor_list = torch.empty((0, embbedding_size), dtype=torch.float32)
    for iter_ in range(0,iter_list_len,loop_split):
        print("sentence embedded", np.round(iter_/iter_list_len,2),"%", end = '')
        iter_list_snap = encode(iter_list[iter_:iter_+loop_split])
        tensor_list = torch.cat((tensor_list, iter_list_snap))
    return tensor_list


# ################################################  split text to Json # ################################################
def split_dict_json(text):

  dict_keys = extract_uppercase_key_phrases(text)
  capture_key = {}

  for key_ in dict_keys:
    capture_key[key_] = extract_text_after(key_, text, stop_words=dict_keys)

  for _key_ in capture_key:
    capture_key[_key_] = corpus_cleaning(capture_key[_key_])

  for _key_ in capture_key:
    capture_key[_key_] = drop_short_strings(capture_key[_key_].split('.'))

  return capture_key

import shutil

def zip_folder(folder_path, zip_filename):
    """
    Zip the contents of a folder using shutil.

    Parameters:
    - folder_path (str): Path to the folder to be zipped.
    - zip_filename (str): Name of the resulting ZIP file.

    Returns:
    - None
    """
    shutil.make_archive(zip_filename, 'zip', folder_path)

# Example usage:
# folder_to_zip = '/content/Bio_ClinicalBERT_tokenizer_reduced_embedding_size'
# zip_file_name = 'Bio_ClinicalBERT_tokenizer_reduced_embedding_size'
# zip_folder(folder_to_zip, zip_file_name)

def unzip_folder(zip_filename, extract_folder):
    """
    Unzip the contents of a ZIP file.

    Parameters:
    - zip_filename (str): Name of the ZIP file to be extracted.
    - extract_folder (str): Folder where the contents will be extracted.

    Returns:
    - None
    """
    shutil.unpack_archive(zip_filename, extract_folder)


################################################################### Extract text from keyword ########################################################################

def extract_text_from_keyword (sample_text, keyword):
    list_of_text = []
    for i in range(len(sample_text)):
        if sample_text[i].find(keyword) != -1:
            list_of_text.append(sample_text[i])
    return list_of_text


################################################################### convert dict to df ########################################################################
def convert_dict_to_df(dict_):
  master_key = []; sub_key = []; sub_key_codes = []; sub_key_descriptions = [];
  for _key_1 in dict_:
    for _key_2 in dict_[_key_1]:
      try:
        master_key.append(_key_1)
      except:
        master_key.append('')
      try:
        sub_key.append(_key_2)
      except:
        sub_key.append('')
      try:
        sub_key_codes.append(dict_[_key_1][_key_2]['CODE'])
      except:
        sub_key_codes.append('')
      try:
        sub_key_descriptions.append(dict_[_key_1][_key_2]['DESCRIPTION'])
      except:
        sub_key_descriptions.append('')

  return_df = pd.DataFrame({
      'CODE_FAMILY': master_key,
      'CODE_SUB_FAMILY': sub_key,
      'CODES': sub_key_codes,
      'DESCRIPTION': sub_key_descriptions
      })
  return return_df


def diag_code_assign (json_input, embedding, dataframe_, key_name = 'DIAGNOSIS', top_rank = 3, threshold = 0.85):
  master_json =  {'CODE': [] ,
                'DESCRIPTION': []}
  for i in range(len(json_input[key_name])):
    input_text = json_input[key_name][i]
    #Encode query and docs
    input_query = encode(input_text)
    #Compute dot score between query and all document embeddings
    match_scores = torch.mm(input_query, embedding.transpose(0, 1))[0].cpu().tolist()
    rank_scores = scipy.stats.rankdata(-np.array(match_scores), method = 'ordinal')
    condition_bool = np.array([True if (rank_scores[i] <= top_rank) & (match_scores[i] > threshold) else False for i in range(len(rank_scores))])
    extracted_df = dataframe_.loc[condition_bool, ['CODE', 'SHORT_DESCRIPTION']]
    master_json = {'CODE': master_json['CODE'] + extracted_df['CODE'].to_list(),
                   'DESCRIPTION': master_json['DESCRIPTION'] + extracted_df['SHORT_DESCRIPTION'].to_list()}
  return master_json



def cpt_code_assign (json_input, embedding, dataframe_, key_name = 'PROCEDURE', top_rank = 3, threshold = 0.85):
  master_json =  {'CODE': [] ,
                'DESCRIPTION': []}
  for i in range(len(json_input[key_name])):
    input_text = json_input[key_name][i]
    #Encode query and docs
    input_query = encode(input_text)
    #Compute dot score between query and all document embeddings
    match_scores = torch.mm(input_query, embedding.transpose(0, 1))[0].cpu().tolist()
    rank_scores = scipy.stats.rankdata(-np.array(match_scores), method = 'ordinal')
    condition_bool = np.array([True if (rank_scores[i] <= top_rank) & (match_scores[i] > threshold) else False for i in range(len(rank_scores))])
    extracted_df = dataframe_.loc[condition_bool, ['cpt_code'	,'description']]
    master_json = {'CODE': master_json['CODE'] + extracted_df['cpt_code'].to_list(),
                   'DESCRIPTION': master_json['DESCRIPTION'] + extracted_df['description'].to_list()}
  return master_json


def diag_and_cpt_tagging(text, diag_embedding, cpt_embedding, diag_df, cpt_df,
                         diag_key = '', cpt_key= '', 
                         diag_rank = 2, cpt_rank = 2, _threshold_ = 0.2, return_dataframe = False ):
  json_input = split_dict_json(text)
  json_input_keys = list(json_input.keys())

  # Diagnosis key list
  if diag_key == '':
    pass
  else:
    diag_key = extract_text_from_keyword(json_input_keys, diag_key)
    # print(diag_key)

  # CPT key list
  if cpt_key == '':
    pass
  else:
    cpt_key = extract_text_from_keyword(json_input_keys, cpt_key)
    # print(cpt_key)

  master_key = {}
  # Diagnosis Mapping
  if diag_key == '':
    pass
  else:
    diag_dict_master = {}
    for _key_ in diag_key:
      diag_dict = diag_code_assign (json_input = json_input, embedding= diag_embedding, dataframe_=diag_df,
                                    key_name = _key_, top_rank = diag_rank, threshold = _threshold_)
      diag_dict_master[_key_] = diag_dict
    master_key['DIAGNOSIS'] = diag_dict_master

  # procedure Mapping
  if cpt_key == '':
    pass
  else:
    cpt_dict_master = {}
    for _key_ in cpt_key:
      cpt_dict = cpt_code_assign (json_input = json_input, embedding= cpt_embedding, dataframe_=cpt_df,
                                    key_name = _key_, top_rank = cpt_rank, threshold = _threshold_)
      cpt_dict_master[_key_] = cpt_dict
    master_key['PROCEDURES'] = cpt_dict_master
  if return_dataframe:
    return convert_dict_to_df(master_key)
  return master_key
