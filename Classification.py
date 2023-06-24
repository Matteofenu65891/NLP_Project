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

def MostSpecifiedTypes(labels):
    from rdflib import Graph, Namespace
    from rdflib.plugins.sparql import prepareQuery
    base_path="http://dbpedia.org/ontology/"
    # Carica l'ontologia DBpedia in un grafo RDF
    g = Graph()
    g.parse(r"C:\Users\fenum\Desktop\progetto nlp\NLP_Project\ontology--DEV_type=orig.owl",
            format="xml")

    # Definisci il namespace per le risorse di DBpedia
    dbpedia = Namespace("http://dbpedia.org/ontology/")

    # Esempio di lista di tipi da valutare
    input_types=[]
    for name in labels:
        input_types.append(base_path+name)

    # Trova il tipo più specifico nella lista di tipi
    most_specific_type = None
    max_specificity = float('-inf')

    # Esegui una query SPARQL per ottenere la specificità dei tipi
    query = prepareQuery(
        """
        SELECT ?type
        WHERE {
            VALUES ?type { %s }
            { ?type rdfs:subClassOf+ ?superType }
            UNION
            { ?superType rdfs:subClassOf+ ?type }
            FILTER NOT EXISTS { ?type rdfs:subClassOf ?otherType }
        }
        """ % (" ".join("<%s>" % t for t in input_types)),
        initNs={"dbpedia": dbpedia, "rdfs": g.namespace_manager}
    )

    # Esegui la query per ottenere i risultati
    results = g.query(query)

    # Trova il tipo più specifico
    for row in results:
        current_type = row["type"].toPython()
        current_specificity = input_types.index(current_type)

        if current_specificity > max_specificity:
            max_specificity = current_specificity
            most_specific_type = current_type

    return most_specific_type