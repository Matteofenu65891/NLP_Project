from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.svm import SVC
from rdflib import Graph, Namespace
from rdflib.plugins.sparql import prepareQuery

def CreateFit(x):
    vectorizer=CountVectorizer()
    fit=vectorizer.fit([item for item in x])
    return fit

def NaiveBayesClassification(x,y,fit):
    bag_of_words = fit.transform([item for item in x])
    x=bag_of_words.toarray()

    y=np.array(list(y))
    clf=MultinomialNB()
    return clf.fit(x,y)

def PenalizedSVM(dict_forLabel, fit):
    bag_of_words = fit.transform([dict_forLabel[n] for n in list(dict_forLabel.keys())])
    x = bag_of_words.toarray()

    y = np.array(list(dict_forLabel.keys()))

    svc_model=SVC(class_weight='balanced', probability=True)
    svc_model.fit(x,y)

    return svc_model
