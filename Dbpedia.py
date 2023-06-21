import json
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import PreProcessing as pr
import Question
import Classification as cl
import Test


if __name__ == '__main__':
    print('ciao')
    f = open(r"smart-2022-datasets-main\AT_answer_type_prediction\dbpedia\SMART2022-AT-dbpedia-train.json")
    data=json.load(f) #json originale sempre utilizzato


    #CAMPIONAMENTO
    #data=(pr.DistinctLabel(data))
    #df = pd.DataFrame.from_dict(data, orient='columns')
    #pr.Sampling(df.values)

    dict_forLabel = pr.createCorpora(data)

    fit = cl.CreateFit(dict_forLabel)
    model = cl.NaiveBayesClassification(dict_forLabel,fit)

    num_corretti=Test.TestNaiveBayes(data, model, fit, 100)
    #num_corretti=Test.TestSVC(data,dict_forLabel,100,fit)
    print(str(num_corretti/len(data)))






