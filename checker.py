import utils_pset
import solutions

def check_solution(answer, reference, mapping,threshold=0.0001):

    gt = utils_pset.Ground_truth()
    for i in range(5,25,5):
        data = gt.run_program(i)
        gt = reference(data,mapping)
        student_answer = answer(data,mapping)
        if abs(gt-student_answer) > threshold:
            return False
    return True

