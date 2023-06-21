import Classification as cl
import PreProcessing as pr

def TestNaiveBayes(data, model, fit, max_number_items):
    num_corretti = 0
    list_errori = {}
    for item in data:
        sentences = [item['question']]
        new_obs = fit.transform(sentences)
        pred = model.predict(new_obs)[0]
        sentences = [pr.PreProcessing(item['question'])]

        skip_prediction = False

        tok = pr.tokenization(item['question'])
        if (tok[0] in pr.auxiliary_verbs):
            pred = 'boolean'
            skip_prediction = True

        if not skip_prediction:
            new_obs = fit.transform(sentences)
            pred = model.predict(new_obs)[0]

        if pred in item['type']:
            num_corretti += 1
        else:
            list_errori[item["question"]] = (pred, item['type'])
            list_errori[item['question']] = (pred, item['type'])

    return num_corretti

def TestSVC(data,dict_for_Label, max_number_items, fit):
    model=cl.PenalizedSVM(dict_for_Label, fit)
    num_corretti=0
    list_errori={}
    for item in data[0:max_number_items]:
        sentences = [pr.PreProcessing(item['question'])]
        new_obs = fit.transform(sentences).toarray()
        pred = model.predict(new_obs)[0]

        if pred in item['type']:
            num_corretti += 1
        else:
            list_errori[item["question"]] = (pred, item['type'])
            list_errori[item['question']] = (pred, item['type'])
    return num_corretti


