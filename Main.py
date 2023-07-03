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
import evaluator
from sklearn.preprocessing import MultiLabelBinarizer
import Classification as cl
pd.options.mode.chained_assignment = None
import pickle
nltk.download('punkt')

def getRawDatasetFromFile(url):
    f = open(url)
    data = json.load(f)  # json originale sempre utilizzato
    return pd.DataFrame.from_dict(data, orient='columns')


def PredictAnswers(answer,model):
    text=pr.CleanText(answer)
    vectorizer = pickle.load(open("Vectorizer/vectorizer.pickle", 'rb'))
    text_to_predict= vectorizer.transform([text])
    y_pred=model.predict(text_to_predict)[0]

    return y_pred

def PredictAllTestSet(dataset_test, model):

    gold_answers={}
    sys_answers={}
    for index, record in dataset_test.iterrows():
        prediction=PredictAnswers(dataset_test.question[index], model)
        sys_answers[dataset_test.id[index]]=prediction
        gold_answers[dataset_test.id[index]] = dataset_test.type[index]

    return gold_answers,sys_answers

def saveModel(model,filename):
    pickle.dump(model, open("Models/"+filename, 'wb'))
    print("Modello salvato")

def loadModel(filename):
    loaded_model = pickle.load(open("Models/"+filename, 'rb'))
    return loaded_model

if __name__ == '__main__':

    """#CODICE PER ELABORARE IL DATASET"""
    dataset = getRawDatasetFromFile(r"Dataset\SMART2022-AT-dbpedia-train.json")

    #solo una prova, splitto a metà il dataset e ne uso metà per indurre il modello e metà per predire. Da togliere
    split_point = (len(dataset) // 4)*3
    dataset_train= dataset.iloc[:split_point]
    dataset_test= dataset.iloc[split_point+1:]


    #X, Y = pr.ProcessDataset(dataset_train) #BLOCCO che si occupa di fare il pre-processing
                                                            #delle domande e restituisce i tipi specifici per le label
    #model=cl.LinearSVCModel(X,Y)

    filename = 'linearSVC.sav'
    # save the model to disk
    #saveModel(model, filename)

    #load model from disk
    model=loadModel(filename)


    gold_answers, sys_answers=PredictAllTestSet(dataset_test,model) #BLOCCO che si occupa di predire tutto il test set
    #gold_answers ->dizionario(id_domanda, lista di tipi corretti)
    #sys_answers ->dizionario(id_domanda, tipo predetto dal modello)

    #print(gold_answers)
    #print(sys_answers)



    #TODO: EVALUATION


    print("Inizio a valutare")
    precisione,recall,f1=evaluator.evaluate_dbpedia(gold_answers, sys_answers)
    print("precisione:"+str(precisione))
    print("recall:"+str(recall))
    print("f1"+str(f1))
    #accuracy = accuracy_score(testset.type, y_pred)
    #accuracy_percent = accuracy * 100
    #print('accuracy %s' % accuracy_percent)







