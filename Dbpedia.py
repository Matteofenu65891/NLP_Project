import json
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import PreProcessing as pr

def createCorpora(data):
    result = {}
    for item in data:
        for type in item['type']:
            if type in result:
                result[type] = result[type]+" "+item['question']
            else:
                result[type] = item['question']

    for el in result:
        result[el]=pr.PreProcessing(result[el])

    return result

def CreateFit(dict_for_label):
    vectorizer=CountVectorizer()
    fit=vectorizer.fit([dict_forLabel[n] for n in list(dict_forLabel.keys())])
    return fit

def NaiveBayesClassification(dict_forLabel,fit):
    bag_of_words = fit.transform([dict_forLabel[n] for n in list(dict_forLabel.keys())])
    x=bag_of_words.toarray()

    y=np.array(list(dict_forLabel.keys()))
    clf=MultinomialNB()
    return clf.fit(x,y)

if __name__ == '__main__':
    f = open(r"smart-2022-datasets-main\AT_answer_type_prediction\dbpedia\SMART2022-AT-dbpedia-train.json")
    data = json.load(f)

    dict_forLabel = createCorpora(data)
    print(list(dict_forLabel.keys()))

    fit=CreateFit(dict_forLabel)
    model=NaiveBayesClassification(dict_forLabel,fit)

    num_corretti=0
    list_errori={}
    for item in data[0:100]:
        sentences=[pr.PreProcessing(item['question'])]

        skip_prediction=False

        tok=pr.tokenization(item['question'])
        if(tok[0] in pr.auxiliary_verbs):
            pred='boolean'
            skip_prediction=True

        if not skip_prediction:
            new_obs = fit.transform(sentences)
            pred=model.predict(new_obs)[0]

        if pred in item['type']:
            num_corretti += 1
        else:
            list_errori[item['question']]=(pred,item['type'])

    print(str(num_corretti/100))
    print(list_errori)
    f.close()






