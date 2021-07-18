import gensim
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import PunktSentenceTokenizer
from bert_serving.client import BertClient
bc = BertClient()
import numpy as np

def reminder(clean_reviews):
  words = []

  for sentence in clean_reviews:
    temp = nltk.word_tokenize(sentence)
    words.extend(temp)
  combined_sentence = [""]
  combined_sentence[0] = " ".join(words)
  words_vec = bc.encode(words)
  sent_vec = bc.encode(combined_sentence)
  similarities = cosine_similarity(sent_vec, words_vec)[0]

  sorted_list = np.argsort(similarities)[::-1]
  words_bert = []

  for k in sorted_list:
    if words[k] in words_bert or len(words_bert) >= 20:
      continue
    words_bert.append(words[k])
    
  words_tfidf = []

  vectorizer = TfidfVectorizer()

  X = vectorizer.fit_transform(clean_reviews)
  X = np.array(X.todense())
  X = np.mean(X, axis = 0)

  X = np.argsort(X)[::-1]

  temp = vectorizer.get_feature_names()

  for i in X:
    words_tfidf.append(temp[i])
  clean_reviews_tokens = []

  for i in clean_reviews:
    clean_reviews_tokens.append(i.split())

  dictionary = gensim.corpora.Dictionary(clean_reviews_tokens)
  bow_corpus = [dictionary.doc2bow(doc) for doc in clean_reviews_tokens]

  lda_model =  gensim.models.LdaMulticore(bow_corpus, num_topics = 10, id2word = dictionary, passes = 10, workers = 2)
  words_lda = []

  for idx, topic in lda_model.print_topics(-1):
      temp = topic.split('"')
      for i in range(len(temp)):
        if i%2 == 0 or temp[i] in words_lda:
          continue
        words_lda.append(temp[i])
  final_list = list(set(words_bert) & set(words_tfidf) & set(words_lda))

  return final_list