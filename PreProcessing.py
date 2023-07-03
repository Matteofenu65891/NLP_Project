import json
import pickle
import re
import nltk
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import Specificita as sp
from nltk.stem import PorterStemmer
import pandas as pd



#CONTSTANT
auxiliary_verbs = ['is', 'am', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                   'did', 'can', 'could',
                   'shall', 'should', 'will', 'would', 'may', 'might', 'must', 'dare', 'need', 'used to', 'ought to']
stopwords = set(STOP_WORDS)
RE_SPECIAL_CHAR = re.compile('[/(){}\[\]|@,;?!.]')
RE_BAD_SYMBOLS = re.compile('[^0-9a-z #+_]')
CONTRACTION_DICT = {"ain't": "are not","'s":" is","aren't": "are not"}

#REGEX
contractions_re=re.compile('(%s)' % '|'.join(CONTRACTION_DICT.keys()))

#<editor-fold desc="clean text">

def TokenizeQuestions(dataset):
    for index, record in dataset.iterrows():
        sentence=dataset.question[index]
        dataset.question[index] = word_tokenize(sentence)

#Procedure for expand contractions
def expand_contractions(text,contractions_dict=CONTRACTION_DICT):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

#Stemming
def stem_words(text):
    return " ".join([PorterStemmer.stem(word) for word in text.split()])

#Full pre-processing single question: expand contractions, lower case, remove punctuations, remove words and digit containing digits, remove stop words, stemming
def CleanText(text):
    text = text.lower()
    text = RE_SPECIAL_CHAR.sub(' ', text)  # sostituiamo caratteri speciali con spazi
    text = RE_BAD_SYMBOLS.sub(' ', text)
    text=re.sub('W*dw*',' ',text)
    #text = ' '.join(word for word in text.split() if word not in STOP_WORDS)  # if word not in STOP_WORDS
    text=expand_contractions(text)
    return text

#</editor-fold>

#<editor-fold desc="utils">
def salvaDatasetLavoratoSuFile(dataset,url):
    dataset.to_csv(url, index=False)
    print("salvati su file")

def leggiDatasetDaFile(url):
    df = pd.read_csv(url)
    return df

def sorter(e):
    return e['num']
#</editor-fold>

#<editor-fold desc="pruining label inconsistenti">
def GetDictionaryOfTypesWithFrequency(types):
    result={}
    for element in types:
        for type in element:
            if type in result:
                result[type]+=1
            else:
                result[type]=1

    for el in result:
        result[el]=(result[el]/len(types))*100

    return result

def pruningLabelInconsistenti(dataset,treshold):
    # Calcolo della frequenza per ogni possibile label di classe
    dict_l = GetDictionaryOfTypesWithFrequency(dataset.type)  # dizionario tipo,frequenza
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


#</editor-fold>


#<editor-fold desk="ricerca label specifiche">
#Funzione che prende una lista di tipi e restituisce il più specifico in base alla specificità dettata dall'ontologia
#Necessita di un file contenente l'ontologia elaborata (far partire il main nel file Specificita per elaborare il file passato tramite percorso)
def getTipoSpecifico(tipi,type_hierarchy):


    urlTipi=[]
    for tipo in tipi:
        if(tipo != "number" and tipo != "string" and tipo != "date" and tipo != "boolean"):
            tipo = tipo[4:]
        urlTipi.append("http://dbpedia.org/ontology/"+tipo)
    orderedList = sp.sort_types(urlTipi,type_hierarchy)
    #print(orderedList)
    orderedList.sort(key=sorter,reverse=True)
    mostSpecific=orderedList[0]['key']
    mostSpecific=mostSpecific.split('/')[-1]
    mostSpecific="dbo:"+mostSpecific
    return mostSpecific



def trovaLabelSpecifiche(dataset):
    #generare la type_hierarchy dallo script Specificita
    with open("Ontology/orderedTypeList", "r") as fp:
        type_hierarchy = json.load(fp)
    for index, record in dataset.iterrows():
        dataset.id[index] = index
        dataset.type[index] = getTipoSpecifico(record.type, type_hierarchy)
    return dataset

#</editor-fold>

def ProcessDataset(dataset):
    for index, record in dataset.iterrows():
        dataset.question[index] = CleanText(record.question)
    vectorizer = TfidfVectorizer()
    dataset_questions = vectorizer.fit_transform(dataset["question"].values)
    pickle.dump(vectorizer, open("Vectorizer/vectorizer.pickle", "wb"))

    # pruining label inconsistenti (non lo facciamo più ma l'ho messo)
    # dataset=pruningLabelInconsistenti(dataset,1.5)

    dataset = trovaLabelSpecifiche(dataset)

    print("pre-processing terminato")

    return (dataset_questions, dataset.type)


