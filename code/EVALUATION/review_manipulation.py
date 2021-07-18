from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')

import os
import pandas as pd
from nltk import sent_tokenize
count = 1

def extract_imperatives(reviews): #it takes a list of reviews and returns a list of imperatives
    imperatives = []
    
    for review in reviews:
        
        sentences = sent_tokenize(review)
    
        for sent in sentences: 
            result = nlp.annotate(sent,
                            properties={
                                'annotators': 'pos',
                                'outputFormat': 'json',
                                'timeout': 1000,
                            })
            try:
                if "VB" in result["sentences"][0]["tokens"][0]["pos"]:
                    imperatives.append(sent)
                
            except:
                pass
    return imperatives

'''for file_name in os.listdir('EVALUATION/data/MMR'):
    print(count, 'of', len(os.listdir('EVALUATION/data/MMR')))
    count += 1
    if file_name in os.listdir('EVALUATION/data/manipulated'):
        continue
    path = 'EVALUATION/data/MMR/' + file_name
    df = pd.read_csv(path)
    reviews = list(df.iloc[:, 2].values)
    text = []
    for review in reviews:
        l = sent_tokenize(review)
        if len(l) < 2:
            continue
        imp = extract_imperatives([review])
        if len(imp) == 0:
            continue

        text.append(review)

    df1 = pd.DataFrame()
    df1['text'] = text
    df1.to_csv('EVALUATION/data/manipulated/' + file_name, index = False)'''

for file_name in os.listdir('EVALUATION/data/manipulated'):
    print(count, 'of', len(os.listdir('EVALUATION/data/MMR')))
    count += 1
    path = 'EVALUATION/data/manipulated/' + file_name
    df = pd.read_csv(path)
    df1 = pd.DataFrame()
    reviews = list(df.iloc[:, 0])
    if len(reviews) < 35:
        continue
    df1['text'] = reviews[:35]
    df1.to_csv(path, index = False)