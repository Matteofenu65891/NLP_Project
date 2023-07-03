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
import evaluator
from sklearn.preprocessing import MultiLabelBinarizer
import Classification as cl
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

def PredictAnswers(answer,feature_extraction,model):
    text=pr.CleanText(answer)
    text_to_predict= feature_extraction.transform([text])
    y_pred=model.predict(text_to_predict)[0]

    return y_pred

def GetAllSpecificTypes(y_list,threesold):

    result=[]
    for label in y_list:
        type=pr.getTipoSpecifico(label,threesold)
        result.append(type)
    return result

def ProcessDataset(dataset,feature_extraction):
    for index, record in dataset.iterrows():
        dataset.question[index] = pr.CleanText(record.question)

    X = feature_extraction.fit_transform(dataset["question"].values)

    # convert every labels(list of types) in single string
    #types_strings = GetAllSpecificTypes(dataset.type,0.1)
    dataset=trovaLabelSpecifiche(dataset)

    return(X,dataset.type)

def PredictAllTestSet(dataset_test, model, feature_extraction):

    gold_answers={}
    sys_answers={}
    for index, record in dataset_test.iterrows():
        prediction=PredictAnswers(dataset_test.question[index], feature_extraction, model)
        sys_answers[dataset_test.id[index]]=prediction
        gold_answers[dataset_test.id[index]] = dataset_test.type[index]

    return gold_answers,sys_answers

if __name__ == '__main__':

    """#CODICE PER ELABORARE IL DATASET"""
    dataset = getRawDatasetFromFile(r"smart-2022-datasets-main\AT_answer_type_prediction\dbpedia\SMART2022-AT-dbpedia-train.json")

    #solo una prova, splitto a metà il dataset e ne uso metà per indurre il modello e metà per predirre. Da togliere
    half_length = len(dataset) // 2
    dataset_train= dataset.iloc[:half_length]
    dataset_test= dataset.iloc[half_length+1:]

    feature_extraction = TfidfVectorizer()
    X, Y = ProcessDataset(dataset_train,feature_extraction) #BLOCCO che si occupa di fare il pre-processing
                                                            #delle domande e restituisce i tipi specifici per le label
    print("pre-processing terminato")


    model=cl.LinearSVCModel(X,Y)
    ##TODO: SaveModel(), ImportModel(path)##

    gold_answers, sys_answers=PredictAllTestSet(dataset_test,model,feature_extraction) #BLOCCO che si occupa di predirre tutto il test set
    #gold_answers ->dizionario(id_domanda, lista di tipi corretti)
    #sys_answers ->dizionario(id_domanda, tipo predetto dal modello)

    print(gold_answers)
    print(sys_answers)

    #TODO: EVALUATION


    #print("Inizio a valutare")
    #precisione,recall,f1=evaluator.evaluate_dbpedia(gold_answers, sys_answers)
    #print("precisione:"+str(precisione))
    #print("recall:"+str(recall))
    #print("f1"+str(f1))
    #accuracy = accuracy_score(testset.type, y_pred)
    #accuracy_percent = accuracy * 100
    #print('accuracy %s' % accuracy_percent)







