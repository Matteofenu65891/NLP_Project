import json
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import PreProcessing as pr
import Question
import Classification as cl
import Test
from sklearn.metrics import classification_report,accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None

def NB_model(x,y):

    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB()),
                   ])
    nb.fit(x, y)
    return nb

if __name__ == '__main__':
    print('ciao')
    f = open(r"smart-2022-datasets-main\AT_answer_type_prediction\dbpedia\SMART2022-AT-dbpedia-train.json")
    data=json.load(f) #json originale sempre utilizzato
    dataset = pd.DataFrame.from_dict(data, orient='columns')

    i=0
    for el in dataset.type:
        val=""
        for item in el:
            val=val+" "+item
        dataset.type[i]=val
        i+=1

    for item in dataset.question:
        item=pr.CleanText(item)

    X= dataset.question
    y = dataset.type
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    nb=NB_model(X_train,y_train)
    y_pred= nb.predict(X_test)
    print('accuracy %s' % accuracy_score(y_test, y_pred, normalize=False))

    #print(classification_report(y_test, y_pred, target_names=my_tags))
    #CAMPIONAMENTO
    #data=(pr.DistinctLabel(data))
    #df = pd.DataFrame.from_dict(data, orient='columns')
    #pr.Sampling(df.values)

    # dict_forLabel = pr.createCorpora(data)
    #
    # fit = cl.CreateFit(dict_forLabel)
    # model = cl.NaiveBayesClassification(dict_forLabel,fit)
    #
    # num_corretti=Test.TestNaiveBayes(data, model, fit, 100)
    # #num_corretti=Test.TestSVC(data,dict_forLabel,100,fit)
    # print(str(num_corretti/len(data)))






