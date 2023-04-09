import utils_pset
import solutions
import numpy as np

def check_solution(answer, reference, mapping,threshold=0.0001):

    gt = utils_pset.Ground_truth()
    for i in range(5,25,5):
        images, coords, actions, rewards = gt.run_program(N=i)
        frames =utils_pset.raw_to_frame(images, coords, actions, rewards)
        gt_answer = reference(frames,mapping)
        student_answer = answer(frames,mapping)
        assert np.max(np.abs(gt_answer-student_answer)) > threshold


if __name__ == "__main__":
    mapping = np.arange(200).reshape((2, 100))
    print(check_solution(solutions.temporal_cohesion_sol, solutions.temporal_cohesion_sol, mapping))

