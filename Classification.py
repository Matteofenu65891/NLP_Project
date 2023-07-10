from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from rdflib import Graph, Namespace
from rdflib.plugins.sparql import prepareQuery
from sklearn.svm import LinearSVC,SVC
from sklearn.neural_network import MLPClassifier

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

def NB_model(x,y):

    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB()),
                   ])
    nb.fit(x, y)
    return nb

def LinearSupportVectorMachine(X_train,y_train):
    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfVectorizer()),
                    ('clf',
                     SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                    ])
    sgd.fit(X_train, y_train)

    return sgd

def LogisticRegressionModel(X_train, y_train): #il migliore al momento, 71% sul test set e 99% sul training
    logreg = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('clf', LogisticRegression(n_jobs=1, C=1e5)),
                       ])
    logreg.fit(X_train, y_train)
    return logreg

def PenalizedSVM(dict_forLabel, fit):
    bag_of_words = fit.transform([dict_forLabel[n] for n in list(dict_forLabel.keys())])
    x = bag_of_words.toarray()

    y = np.array(list(dict_forLabel.keys()))

    svc_model=SVC(class_weight='balanced', probability=True)
    svc_model.fit(x,y)

    return svc_model

def LinearSVCModel(X,Y):
    model = LinearSVC(tol=1.0e-6, verbose=1)
    model.fit(X, Y)
    print("Modello creato")
    return model

def PolinomialSVCModel(X,Y):
    model = SVC(kernel='poly', degree=3)
    model.fit(X, Y)
    print("Modello creato")
    return model

def NNModel(X,Y):
    NN_model = MLPClassifier(random_state=1, max_iter=300)
    NN_model.fit(X,Y)

    return NN_model