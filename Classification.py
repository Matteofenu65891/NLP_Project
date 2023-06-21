from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.svm import SVC

def CreateFit(dict_forLabel):
    vectorizer=CountVectorizer()
    fit=vectorizer.fit([dict_forLabel[n] for n in list(dict_forLabel.keys())])
    return fit

def NaiveBayesClassification(dict_forLabel,fit):
    bag_of_words = fit.transform([dict_forLabel[n] for n in list(dict_forLabel.keys())])
    x=bag_of_words.toarray()

    y=np.array(list(dict_forLabel.keys()))
    clf=MultinomialNB()
    return clf.fit(x,y)

def PenalizedSVM(dict_forLabel, fit):
    bag_of_words = fit.transform([dict_forLabel[n] for n in list(dict_forLabel.keys())])
    x = bag_of_words.toarray()

    y = np.array(list(dict_forLabel.keys()))

    svc_model=SVC(class_weight='balanced', probability=True)
    svc_model.fit(x,y)

    return svc_model