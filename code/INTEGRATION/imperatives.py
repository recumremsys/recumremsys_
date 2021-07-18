import nltk
import copy
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import PunktSentenceTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from nltk.corpus import stopwords

from bert_serving.client import BertClient
bc = BertClient()

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
stemmer = SnowballStemmer("english")

STOPWORDS = stopwords.words("english")

def extract_imperatives(sample_text, clean):
  custom_sent_tokenizer = PunktSentenceTokenizer()
  tokenized = custom_sent_tokenizer.tokenize(sample_text)

  temp = [""]
  temp1 = []
  for sent in tokenized:
    temp[0] = sent.split()[0]
    if nltk.pos_tag(temp)[0][1] == "VB" :
      temp1.append(sent)
  print(temp1)
  impe_sent = temp1
  if clean == 1:
    for i in range(len(impe_sent)):
      text = impe_sent[i].lower()  # lowercase text
      text = BAD_SYMBOLS_RE.sub(' ', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
      text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwords from text
      text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
      text = ''.join(text)
      impe_sent[i] = text

  ##############################
  document = [""]
  document[0] = " ".join(impe_sent)
  #noun_phrase_encodings is a list of encodings of all the extracted phrases 
  impe_sent_encodings = bc.encode(impe_sent)

  #document_encoding contains encoding of the document
  document_encoding = bc.encode(document)

  similarities = cosine_similarity(document_encoding,impe_sent_encodings)[0]

  #sorted_list is the list of indexes with decreasing cosine values
  sorted_list = np.argsort(similarities)[::-1]

  ###############################
  lambda_score = 0.5
  mmr_list = [sorted_list[0]]
  mmr_scores = [similarities[mmr_list[0]]*lambda_score]
  temp_sorted_list = copy.deepcopy(sorted_list)

  while len(temp_sorted_list) != 1:
    maximum = -10**4
    max_D_i = None

    for D_i in temp_sorted_list:
      if D_i == temp_sorted_list[0]:
        continue
      temp = []
      for j in mmr_list:
        temp.append(impe_sent_encodings[j])
      temp1 = [[]]
      temp1[0] = (impe_sent_encodings[D_i])
      sim2 = max(cosine_similarity(temp1,temp))[0]
      score = lambda_score*(similarities[D_i] - (1 - lambda_score) * sim2)
      if score > maximum:
        maximum = score
        max_D_i = D_i
    mmr_list.append(max_D_i)
    mmr_scores.append(maximum)
    temp_sorted_list = np.delete(temp_sorted_list, np.argwhere(temp_sorted_list==max_D_i))
  
  for i in range(len(mmr_list)):
    mmr_list[i] = impe_sent[mmr_list[i]]

  return mmr_list