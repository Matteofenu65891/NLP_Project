import json
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import PreProcessing as pr
from sklearn.metrics import classification_report,accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
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

if __name__ == '__main__':

    """#CODICE PER ELABORARE IL DATASET
    dataset = getRawDatasetFromFile(r"smart-2022-datasets-main\AT_answer_type_prediction\dbpedia\SMART2022-AT-dbpedia-train.json")
    #pulizia del testo delle domande

    for index, record in dataset.iterrows():
        dataset.question[index]=pr.CleanText(record.question)

    treshold=1.5
    dataset=pruningLabelInconsistenti(dataset,treshold)

    #CAMPIONAMENTO
    dataset.sample()

    dataset=trovaLabelSpecifiche(dataset)
    
    #SALVATAGGIO DATASET ELABORATO SU FILE
    #salvaDatasetLavoratoSuFile(dataset,'dataTypeSpecifici.csv')"""

    #CODICE PER LEGGERE UN DATASET GIA ELABORATO
    dataset=leggiDatasetDaFile('dataTypeSpecifici.csv')

    X_train, X_test, y_train, y_test = train_test_split(dataset.question, dataset.type, test_size=0.33, random_state=42)
    # Creazione dell'istanza del RandomUnderSampler
    #fit=cl.CreateFit(X_train)
    nb=LogisticRegressionModel(X_train,y_train)
    y_pred= nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_percent = accuracy * 100
    print('accuracy %s' % accuracy_percent)







