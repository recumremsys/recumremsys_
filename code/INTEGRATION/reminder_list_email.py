import re
import nltk
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tag.stanford import StanfordPOSTagger
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
path_to_model = "stanford-postagger-2018-10-16/models/english-bidirectional-distsim.tagger"
path_to_jar = "stanford-postagger-2018-10-16/stanford-postagger.jar"
s_pos_tagger=StanfordPOSTagger(path_to_model, path_to_jar)


def reminder_email(text):
  tokenizer = PunktSentenceTokenizer()
  sent = tokenizer.tokenize(text)
  imperative_sentences = []

  for s in sent:
    #s = clean_text(s)
    temp_pos_tag = s_pos_tagger.tag([s.split()[0]])
    if temp_pos_tag[0][1] == "VB" or s.strip().split()[-1][-1] == "!":
      imperative_sentences.append(s)

  nouns = []

  for s in imperative_sentences:
    #s = clean_text(s)
    print(s)
    s = s.split()
    pos_tags = nltk.pos_tag(s)
    nouns.append([])
    for i in pos_tags:
      if i[1] == "NN":
        nouns[-1].append(i)
    #print(pos_tags)
    #print(nouns[-1])
  return nouns