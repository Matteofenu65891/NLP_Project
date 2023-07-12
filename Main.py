import json
import pandas as pd
import PreProcessing as pr
import nltk
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


def PredictAnswers(answer,category,model):
    text=pr.CleanText(answer)
    vectorizer1 = pickle.load(open("Vectorizer/vectorizer1.pickle", 'rb'))
    vectorizer2 = pickle.load(open("Vectorizer/vectorizer2.pickle", 'rb'))
    text_to_predict= vectorizer1.transform([text])
    category=vectorizer2.transform([category])

    X = hstack((text_to_predict, category))
    y_pred=model.predict(X)[0]

    return y_pred

def PredictAnswerNaive(answer, model):
    text = pr.CleanText(answer)
    vectorizer = pickle.load(open("Vectorizer/vectorizerN.pickle", 'rb'))
    text_to_predict = vectorizer.transform([text])

    y_pred = model.predict(text_to_predict)[0]

    return y_pred

def PredictAllTestSet(dataset_test, model):

    gold_answers={}
    sys_answers={}
    quest={}
    for index, record in dataset_test.iterrows():
        prediction=PredictAnswers(dataset_test.question[index],dataset_test.category[index], model)
        sys_answers[dataset_test.id[index]]=prediction
        gold_answers[dataset_test.id[index]] = dataset_test.type[index]
        quest[dataset_test.id[index]]=dataset_test.question[index]
    return gold_answers,sys_answers,quest

def PredictAllTestSetNaive(dataset_test, model):

    gold_answers={}
    sys_answers={}
    quest={}
    for index, record in dataset_test.iterrows():
        prediction=PredictAnswerNaive(dataset_test.question[index], model)
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

    X, Y= pr.ProcessDataset(dataset) #BLOCCO che si occupa di fare il pre-processing
                                             #delle domande e restituisce i tipi specifici per le label

    filename = 'LinearSVC.sav'

    model=cl.LinearSVCModel(X,Y)

    # save the model to disk
    saveModel(model, filename)

    #load model from disk
    #model=loadModel(filename)


    gold_answers, sys_answers,quest=PredictAllTestSet(testset,model) #BLOCCO che si occupa di predire tutto il test set
    print(Evaluation.evaluate_dbpedia(gold_answers,sys_answers,quest))



