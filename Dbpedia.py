import json
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import PreProcessing as pr
import Question
import Classification as cl
import Test
from sklearn.metrics import classification_report,accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from rdflib import Graph
from sklearn.linear_model import SGDClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression

pd.options.mode.chained_assignment = None

def NB_model(x,y):

    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB()),
                   ])
    nb.fit(x, y)
    return nb

def LinearSupportVectorMachine(X_train,y_train):
    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf',
                     SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                    ])
    sgd.fit(X_train, y_train)

    return sgd

def LogisticRegressionModel(X_train, y_train): #il migliore al momento, 61% sul test set e 99% sul training
    logreg = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('clf', LogisticRegression(n_jobs=1, C=1e5)),
                       ])
    logreg.fit(X_train, y_train)
    return logreg

def FindMoreSpecifiedType(chiavi_cercate,dizionario):
    valore_minimo = None
    result=None
    result = min(dizionario.keys() & chiavi_cercate, key=dizionario.get)


    return result

if __name__ == '__main__':

    f = open(r"smart-2022-datasets-main\AT_answer_type_prediction\dbpedia\SMART2022-AT-dbpedia-train.json")
    data=json.load(f) #json originale sempre utilizzato
    dataset = pd.DataFrame.from_dict(data, orient='columns')
    threesold=0.0028

    for item in dataset.question:
        item=pr.CleanText(item)

    #X = dataset.question
    #y = dataset.type
    dict_l= pr.GetDictionaryOfTypesWithFrequency(dataset.type) #dizionario tipo,frequenza
    dict_l=dict(sorted(dict_l.items(), key=lambda x: x[1]))
    elementi_inconsistenti = {key: val for key, val in dict_l.items() if val == 0.002727024815925825}

    for index,records in dataset.iterrows():
        tipi=records.type
        for label in elementi_inconsistenti:
            if label in tipi:
                records.type.remove(label)

    dataset = dataset[dataset['type'].apply(lambda x: len(x) != 0)]
    y = dataset.type
    X=dataset.question
    counter=0


    #per ogni label di classe,tengo il tipo più specifico (frequenza minore)
    for index, value in y.iteritems():
        y[index]=FindMoreSpecifiedType(value,dizionario=dict_l)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # Creazione dell'istanza del RandomUnderSampler
    fit=cl.CreateFit(X_train)
    nb=LogisticRegressionModel(X_train,y_train)
    y_pred= nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_percent = accuracy * 100
    print('accuracy %s' % accuracy_percent)

    #CAMPIONAMENTO
    #data=(pr.DistinctLabel(data))
    #df = pd.DataFrame.from_dict(data, orient='columns')
    #pr.Sampling(df.values)

    # dict_forLabel = pr.createCorpora(data)
    #
    # fit = cl.CreateFit(dict_forLabel)
    # model = cl.NaiveBayesClassification(dict_forLabel,fit)
    #
    # num_corretti=Test.TestNaiveBayes(data, model, fit, 100)
    # #num_corretti=Test.TestSVC(data,dict_forLabel,100,fit)
    # print(str(num_corretti/len(data)))






