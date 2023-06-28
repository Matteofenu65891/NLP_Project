import json
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import PreProcessing as pr
from sklearn.metrics import classification_report,accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
pd.options.mode.chained_assignment = None
nltk.download('punkt')

def NB_model(x,y):

    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB()),
                   ])
    nb.fit(x, y)
    return nb

def LinearSupportVectorMachine(X_train,y_train):
    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfVectorizer()),
                    ('clf',
                     SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                    ])
    sgd.fit(X_train, y_train)

    return sgd

def LogisticRegressionModel(X_train, y_train): #il migliore al momento, 71% sul test set e 99% sul training
    logreg = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('clf', LogisticRegression(n_jobs=1, C=1e5)),
                       ])
    logreg.fit(X_train, y_train)
    return logreg

def getRawDatasetFromFile(url):
    f = open(url)
    data = json.load(f)  # json originale sempre utilizzato
    return pd.DataFrame.from_dict(data, orient='columns')

def trovaLabelSpecifiche(dataset):
    #generare la type_hierarchy dallo script Specificita
    with open("orderedTypeList", "r") as fp:
        type_hierarchy = json.load(fp)
    """i = 0
    for item in dataset.type:
        dataset.id[i]=i
        dataset.type[i] = pr.getTipoSpecifico(item, type_hierarchy)
        i += 1"""
    for index, record in dataset.iterrows():
        dataset.id[index] = index
        dataset.type[index] = pr.getTipoSpecifico(record.type, type_hierarchy)
    return dataset

def salvaDatasetLavoratoSuFile(dataset,url):
    dataset.to_csv(url, index=False)
    print("salvati su file")

def leggiDatasetDaFile(url):
    df = pd.read_csv(url)
    return df
def pruningLabelInconsistenti(dataset,treshold):
    # Calcolo della frequenza per ogni possibile label di classe
    dict_l = pr.GetDictionaryOfTypesWithFrequency(dataset.type)  # dizionario tipo,frequenza
    # ordinamento per frequenza
    dict_l = dict(sorted(dict_l.items(), key=lambda x: x[1]))
    # label con soglia di frequenza inferiore al valore specificato
    elementi_inconsistenti = {key: val for key, val in dict_l.items() if val < treshold}
    # scorro il dataset e elimino le label inconsistenti
    for index, records in dataset.iterrows():
        tipi = records.type
        for label in elementi_inconsistenti:
            if label in tipi:
                records.type.remove(label)
    # elimino gli elementi che hanno label di classe vuota (nessuna label abbastanza frequente)
    dataset = dataset[dataset['type'].apply(lambda x: len(x) != 0)]
    return dataset

def TokenizeQuestions(dataset):
    for index, record in dataset.iterrows():
        sentence=dataset.question[index]
        dataset.question[index] = word_tokenize(sentence)


if __name__ == '__main__':

    """#CODICE PER ELABORARE IL DATASET"""
    dataset = getRawDatasetFromFile(r"smart-2022-datasets-main\AT_answer_type_prediction\dbpedia\SMART2022-AT-dbpedia-train.json")
    testset=getRawDatasetFromFile(r"smart-2022-datasets-main\AT_answer_type_prediction\dbpedia\SMART2022-AT-dbpedia-test-risposte.json")
    #pulizia del testo delle domande

    for index, record in dataset.iterrows():
        dataset.question[index]=pr.CleanText(record.question)
    for index, record in testset.iterrows():
        testset.question[index]=pr.CleanText(record.question)

    treshold=1.5
    dataset=pruningLabelInconsistenti(dataset,treshold)
    testset=pruningLabelInconsistenti(testset,treshold)

    #CAMPIONAMENTO
    dataset.sample()

    dataset=trovaLabelSpecifiche(dataset)
    testset=trovaLabelSpecifiche(testset)
    #TokenizeQuestions(dataset)
    #SALVATAGGIO DATASET ELABORATO SU FILE
    #salvaDatasetLavoratoSuFile(dataset,'dataTypeSpecifici.csv')"""

    #CODICE PER LEGGERE UN DATASET GIA ELABORATO
    #dataset=leggiDatasetDaFile('dataTypeSpecifici.csv')
    feature_extraction = TfidfVectorizer()
    X = feature_extraction.fit_transform(dataset["question"].values)
    Y = feature_extraction.fit_transform(testset["question"].values)

    #X_train, X_test, y_train, y_test = train_test_split(X, dataset.type, test_size=0.15, random_state=42)
    # Creazione dell'istanza del RandomUnderSampler
    #fit=cl.CreateFit(X_train)
    #clf = SVC(probability=True, kernel='rbf')
    #clf.fit(X_train, y_train)
    model = LinearSVC(tol=1.0e-6, verbose=1)
    #model.fit(X_train, y_train)
    model.fit(X, dataset.type)
    #nb = LogisticRegressionModel(X_train, y_train)
    y_pred= model.predict(Y)
    accuracy = accuracy_score(testset.type, y_pred)
    accuracy_percent = accuracy * 100
    print('accuracy %s' % accuracy_percent)







