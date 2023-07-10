import json
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import PreProcessing as pr
from sklearn.metrics import classification_report,accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
import numpy as np
from sklearn.svm import SVC
import Evaluation
from sklearn.preprocessing import MultiLabelBinarizer
import Classification as cl
pd.options.mode.chained_assignment = None
import pickle
import Evaluation
from scipy.sparse import hstack
from imblearn.over_sampling import SMOTE
nltk.download('punkt')

def getRawDatasetFromFile(url):
    f = open(url)
    data = json.load(f)  # json originale sempre utilizzato
    return pd.DataFrame.from_dict(data, orient='columns')


def PredictAnswers(answer,category,model,vectorizer1,vectorizer2):
    text=pr.CleanText(answer)
    vectorizer = pickle.load(open("Vectorizer/vectorizer.pickle", 'rb'))
    text_to_predict= vectorizer1.transform([text])
    category=vectorizer2.transform([category])

    X = hstack((text_to_predict, category))
    y_pred=model.predict(X)[0]

    return y_pred

def PredictAnswers2(answer,category,model,vectorizer1,vectorizer2):
    text=pr.CleanText(answer)
    vectorizer = pickle.load(open("Vectorizer/vectorizer.pickle", 'rb'))
    text_to_predict= vectorizer1.transform([text])
    category=vectorizer2.transform([category])

    nuova_domanda_encoded = pd.DataFrame(text_to_predict.toarray())
    nuova_categoria_encoded = pd.Series(category)
    nuovi_dati_encoded = pd.concat([nuova_domanda_encoded, nuova_categoria_encoded], axis=1)
    y_pred=model.predict(nuovi_dati_encoded)[0]

    return y_pred

def PredictAllTestSet(dataset_test, model,vectorizer1,vectorizer2):

    gold_answers={}
    sys_answers={}
    quest={}
    for index, record in dataset_test.iterrows():
        prediction=PredictAnswers2(dataset_test.question[index],dataset_test.category[index], model,vectorizer1,vectorizer2)
        sys_answers[dataset_test.id[index]]=prediction
        gold_answers[dataset_test.id[index]] = dataset_test.type[index]
        quest[dataset_test.id[index]]=dataset_test.question[index]
    return gold_answers,sys_answers,quest

def saveModel(model,filename):
    pickle.dump(model, open("Models/"+filename, 'wb'))
    print("Modello salvato")

def loadModel(filename):
    loaded_model = pickle.load(open("Models/"+filename, 'rb'))
    return loaded_model



if __name__ == '__main__':

    """#CODICE PER ELABORARE IL DATASET"""
    dataset = getRawDatasetFromFile(r"Dataset\SMART2022-AT-dbpedia-train.json")
    testset = getRawDatasetFromFile(r"Dataset\SMART2022-AT-dbpedia-test-risposte.json")

    X, Y ,vectorizer, label_encoder= pr.ProcessDatasetForRegression(dataset) #BLOCCO che si occupa di fare il pre-processing
                                             #delle domande e restituisce i tipi specifici per le label

    filename = 'LRModel.sav'

    model=cl.LogisticRegressionModel(X,Y)

    # save the model to disk
    saveModel(model, filename)

    #load model from disk
    model=loadModel(filename)


    gold_answers, sys_answers,quest=PredictAllTestSet(testset,model,vectorizer,label_encoder) #BLOCCO che si occupa di predire tutto il test set
    print(Evaluation.evaluate_dbpedia(gold_answers,sys_answers,quest))
    #gold_answers ->dizionario(id_domanda, lista di tipi corretti)
    #sys_answers ->dizionario(id_domanda, tipo predetto dal modello)

    #print(gold_answers)
    #print(sys_answers)


