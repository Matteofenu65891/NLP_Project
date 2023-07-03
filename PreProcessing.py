import json
import re
import nltk
from spacy.lang.en.stop_words import STOP_WORDS
from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import Specificita as sp
from nltk.stem import PorterStemmer


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

#full pre-processing dataset
def PreProcessingFullTrainSet(dataset):
    #puliza delle domande
    for index, record in dataset.iterrows():
        dataset.question[index]=pr.CleanText(record.question)


#Procedure for expand contractions
def expand_contractions(text,contractions_dict=CONTRACTION_DICT):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

#Stemming
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

#Full pre-processing single question: expand contractions, lower case, remove punctuations, remove words and digit containing digits, remove stop words, stemming
def CleanText(text):
    text = text.lower()
    text = RE_SPECIAL_CHAR.sub(' ', text)  # sostituiamo caratteri speciali con spazi
    text = RE_BAD_SYMBOLS.sub(' ', text)
    text=re.sub('W*dw*',' ',text)
    text = ' '.join(word for word in text.split() if word not in STOP_WORDS)  # if word not in STOP_WORDS
    text=expand_contractions(text)
    return text

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

def sorter(e):
    return e['num']

