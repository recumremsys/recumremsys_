import nltk
nltk.download('punkt')
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

st = StanfordNERTagger('stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz',
					   'stanford-ner-2018-10-16/stanford-ner.jar',
					   encoding='utf-8')

def NER(text):
  tokenized_text = word_tokenize(text)
  classified_text = st.tag(tokenized_text)
  dic = dict()
  for i in classified_text:
    if i[1] == "O":
      continue
    if i[1] not in dic:
      dic[i[1]] = [i[0]]
      continue
    dic[i[1]].append(i[0])

  return dic