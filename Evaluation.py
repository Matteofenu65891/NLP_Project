#gold_answers: dict(id_answer, list_of_type)
#sys_answers: dict(id_answer, more_specific_type)
def evaluate_dbpedia(gold_answers, system_answers):
    count, total_p, total_r, total_f1 = 0, 0, 0, 0
    for ques_id in gold_answers:
        count += 1
        # if an answer is not provided to a question, we just move on
        if ques_id not in system_answers:
            continue

        gold_answer_list = gold_answers[ques_id] #a list
        prediction = system_answers[ques_id] #an object

        if prediction in gold_answer_list:
            total_p+=1

    return total_p/count
