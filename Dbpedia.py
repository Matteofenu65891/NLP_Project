import json
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import PreProcessing as pr
import Question

def createCorpora(data):
    result = {}
    min=5000

    for item in data:
        key=""
        for type in item['type']:
            key=key+" "+type
        if key in result:
            result[key] = result[key]+" "+item['question']
        else:
            result[key] = item['question']

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

        pred_vet=pred.split(" ")
        correct=True

        pred_vet.remove(pred_vet[0])
        for s in pred_vet:
            if not s in item['type']:
                correct=False

        if correct:
            num_corretti += 1
        else:
            list_errori[item['question']]=(pred,item['type'])

    print(str(num_corretti/100))
    print(list_errori)
    f.close()






