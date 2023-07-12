from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC



def NaiveBayesClassification(x,y):
    clf=MultinomialNB()
    return clf.fit(x,y)


def LogisticRegressionModel(X_train, y_train): #il migliore al momento, 71% sul test set e 99% sul training
        logreg=LogisticRegression(solver='saga',multi_class='multinomial',tol=0.1,penalty='l1')
        logreg.fit(X_train, y_train)
        return logreg

def LinearSVCModel(X,Y):
    model = LinearSVC(tol=1.0e-6, verbose=1)
    model.fit(X, Y)
    print("Modello creato")
    return model

def SGDModel(X_train,y_train):
    clf = SGDClassifier(loss="hinge", penalty="l2")
    clf.fit(X_train, y_train)

    return clf