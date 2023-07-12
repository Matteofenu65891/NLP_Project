#gold_answers: dict(id_answer, list_of_type)
#sys_answers: dict(id_answer, more_specific_type)
literal=['dbo:date','dbo:boolean',"dbo:string","dbo:number"]

#Compute the accuracy of the model
def evaluate_dbpedia(gold_answers, system_answers,quest):
    count, total_p, total_r, total_f1 = 0, 0, 0, 0
    for ques_id in system_answers:
        count += 1
        # if an answer is not provided to a question, we just move on
        if ques_id not in system_answers:
            continue

        gold_answer_list = gold_answers[ques_id] #a list
        prediction = system_answers[ques_id] #an object
        question=quest[ques_id]

        if prediction in literal:
            prediction=prediction.replace("dbo:","")

        if prediction in gold_answer_list:
            total_p+=1
        else:
            print("Prezione su domanda:"+question+" "+prediction)
            print(gold_answer_list)

    print("Predizioni corrette"+str(total_p)+"Su "+str(count))
    return total_p/count


#compute precisione, recall and f-measure for a certain list of types
def ValuateType(gold_answers, system_answers,types):
    for type in types:
        count, tp, fp, fn=0,0,0,0
        for ques_id in system_answers:
            count += 1
            # if an answer is not provided to a question, we just move on
            if ques_id not in system_answers:
                continue

            gold_answer_list = gold_answers[ques_id]  # a list
            prediction = system_answers[ques_id]  # an object

            if prediction in literal:
                prediction = prediction.replace("dbo:", "")

            if prediction in gold_answer_list and prediction == type:
                tp+=1

            if prediction not in gold_answer_list and prediction == type:
                fp+=1

            if type in gold_answer_list and prediction != type:
                fn+=1

        precision,recall,f_measure=0,0,0
        if tp!=0 or fp!=0:
            precision=tp/(tp+fp)

        if tp != 0 or fn != 0:
            recall=tp/(tp+fn)

        if precision!=0 or recall!=0:
            f_measure=2*(precision*recall)/(precision+recall)

        print("Valutazione per il tipo "+type)
        print("recall: "+str(recall))
        print("precision: " + str(precision))
        print("f-measure: " + str(f_measure))
        print(" - - - - - - - - - - - - - - - - - - - - - - ")