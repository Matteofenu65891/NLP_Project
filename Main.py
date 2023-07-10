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
<<<<<<< Updated upstream
=======
>>>>>>> Stashed changes
nltk.download('punkt')

def getRawDatasetFromFile(url):
    f = open(url)
    data = json.load(f)  # json originale sempre utilizzato
    return pd.DataFrame.from_dict(data, orient='columns')


    text=pr.CleanText(answer)
<<<<<<< Updated upstream
    vectorizer = pickle.load(open("Vectorizer/vectorizer.pickle", 'rb'))
    text_to_predict= vectorizer.transform([text])
    y_pred=model.predict(text_to_predict)[0]
=======
    #vectorizer = pickle.load(open("Vectorizer/vectorizer.pickle", 'rb'))

>>>>>>> Stashed changes
    return y_pred


    gold_answers={}
    sys_answers={}
    for index, record in dataset_test.iterrows():
        sys_answers[dataset_test.id[index]]=prediction
        gold_answers[dataset_test.id[index]] = dataset_test.type[index]

def saveModel(model,filename):
    pickle.dump(model, open("Models/"+filename, 'wb'))
    print("Modello salvato")

def loadModel(filename):
    loaded_model = pickle.load(open("Models/"+filename, 'rb'))
    return loaded_model



if __name__ == '__main__':

    """#CODICE PER ELABORARE IL DATASET"""
    #dataset = getRawDatasetFromFile(r"Dataset\SMART2022-AT-dbpedia-train.json")
    dataset = getRawDatasetFromFile(r"Dataset\smarttask_dbpedia_train.json")
    testset = getRawDatasetFromFile(r"Dataset\SMART2022-AT-dbpedia-test-risposte.json")


<<<<<<< Updated upstream
    filename = 'linearSVC2.sav'
=======

>>>>>>> Stashed changes
    # save the model to disk
    saveModel(model, filename)

    #load model from disk
    #model=loadModel(filename)


    #gold_answers ->dizionario(id_domanda, lista di tipi corretti)
    #sys_answers ->dizionario(id_domanda, tipo predetto dal modello)

    #print(gold_answers)
    #print(sys_answers)



    #TODO: EVALUATION


<<<<<<< Updated upstream
    print("Inizio a valutare")
    precisione=Evaluation.evaluate_dbpedia(gold_answers, sys_answers)
    print("precisione:"+str(precisione))
=======
    #accuracy = accuracy_score(testset.type, y_pred)
    #accuracy_percent = accuracy * 100
    #print('accuracy %s' % accuracy_percent)







>>>>>>> Stashed changes
