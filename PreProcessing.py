import re
import nltk
from spacy.lang.en.stop_words import STOP_WORDS
from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder

auxiliary_verbs = ['is', 'am', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                   'did', 'can', 'could',
                   'shall', 'should', 'will', 'would', 'may', 'might', 'must', 'dare', 'need', 'used to', 'ought to']

stopwords = set(STOP_WORDS)
RE_SPECIAL_CHAR = re.compile('[/(){}\[\]\|@,;]')
RE_BAD_SYMBOLS = re.compile('[^0-9a-z #+_]')


# puliamo il testo da caratteri speciali, altri caratteri
def CleanText(text):
    text = text.lower()
    text = RE_SPECIAL_CHAR.sub(' ', text)  # sostituiamo caratteri speciali con spazi
    text = RE_BAD_SYMBOLS.sub(' ', text)
    text = ' '.join(word for word in text.split())  # if word not in STOP_WORDS

    return text

# Per rimuovere le etichette di tipo duplicate
def RemoveDuplicatesType(data):
    for item in data:
        vet_type = item["type"]
        item["type"] = list(set(vet_type))

    return data

def PreProcessing(text):
    output = " "
    word_list2 = tokenization(text)
    for words in word_list2:
        if not words in stopwords and not words in auxiliary_verbs:
            output = output + " " + words

    return output

def DistinctLabel(data):
    result=[]
    for item in data:
        for type in item['type']:
            new_item=item.copy()
            new_item['type']=[type]
            result.append(new_item)
    return result

def Sampling(data):
    # split into input and output elements
    X, y = data[:, :-1], data[:, -1]
    # label encode the target variable
    y = LabelEncoder().fit_transform(y)
    # transform the dataset
    strategy = {0: 100, 1: 100, 2: 200, 3: 200, 4: 200, 5: 200}
    oversample = SMOTE(sampling_strategy=strategy)
    X, y = oversample.fit_resample(X, y)
    # summarize distribution
    counter = Counter(y)
    for k, v in counter.items():
        per = v / len(y) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
    # plot the distribution
    pyplot.bar(counter.keys(), counter.values())
    pyplot.show()

def tokenization(text):
    text = text.lower()
    word_list2 = re.findall(r'\w+', text)

    return word_list2

def createCorpora(data):
    result = {}
    for item in data:
        for type in item['type']:
            if type in result:
                result[type] = result[type]+" "+item['question']
            else:
                result[type] = item['question']

    max_len=1000
    min_len=1000

    for type in result:
        if len(result[type]) < min_len:
            min_len=len(result[type])
        if len(result[type]) > max_len:
            max_len = len(result[type])

    for type in result:
        if not len(result[type])<500:
            result[type]=result[type][:500]
    return result
