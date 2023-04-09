import utils_pset
import solutions
import numpy as np
import solution_vals

def temporal_sol_check(answer):
    mapping = np.arange(200).reshape((2, 100)) / 200
    gt = utils_pset.Ground_truth()

    images, coords, actions, rewards = gt.run_program(N=5)
    frames = utils_pset.raw_to_frame(images, coords, actions, rewards)
    student_answer = answer(frames, mapping)
    assert np.max(np.abs(student_answer - solution_vals.temporal_solution5))

    images, coords, actions, rewards = gt.run_program(N=10)
    frames = utils_pset.raw_to_frame(images, coords, actions, rewards)
    student_answer = answer(frames, mapping)
    assert np.max(np.abs(student_answer - solution_vals.temporal_solution10))

    images, coords, actions, rewards = gt.run_program(N=15)
    frames = utils_pset.raw_to_frame(images, coords, actions, rewards)
    student_answer = answer(frames, mapping)
    assert np.max(np.abs(student_answer - solution_vals.temporal_solution15))

    images, coords, actions, rewards = gt.run_program(N=20)
    frames = utils_pset.raw_to_frame(images, coords, actions, rewards)
    student_answer = answer(frames, mapping)
    assert np.max(np.abs(student_answer - solution_vals.temporal_solution20))

def causal_sol_check(answer):
    mapping = np.arange(200).reshape((2, 100)) / 200
    gt = utils_pset.Ground_truth()

    images, coords, actions, rewards = gt.run_program(N=5)
    frames = utils_pset.raw_to_frame(images, coords, actions, rewards)
    student_answer = answer(frames, mapping)
    assert np.max(np.abs(student_answer - solution_vals.causal_solution5))

    images, coords, actions, rewards = gt.run_program(N=10)
    frames = utils_pset.raw_to_frame(images, coords, actions, rewards)
    student_answer = answer(frames, mapping)
    assert np.max(np.abs(student_answer - solution_vals.causal_solution10))

    images, coords, actions, rewards = gt.run_program(N=15)
    frames = utils_pset.raw_to_frame(images, coords, actions, rewards)
    student_answer = answer(frames, mapping)
    assert np.max(np.abs(student_answer - solution_vals.causal_solution15))

    images, coords, actions, rewards = gt.run_program(N=20)
    frames = utils_pset.raw_to_frame(images, coords, actions, rewards)
    student_answer = answer(frames, mapping)
    assert np.max(np.abs(student_answer - solution_vals.causal_solution20))


def repeatability_sol_check(answer):
    mapping = np.arange(200).reshape((2, 100)) / 200
    gt = utils_pset.Ground_truth()

    images, coords, actions, rewards = gt.run_program(N=5)
    frames = utils_pset.raw_to_frame(images, coords, actions, rewards)
    student_answer = answer(frames, mapping)
    assert np.max(np.abs(student_answer - solution_vals.repeatability_solution5))

    images, coords, actions, rewards = gt.run_program(N=10)
    frames = utils_pset.raw_to_frame(images, coords, actions, rewards)
    student_answer = answer(frames, mapping)
    assert np.max(np.abs(student_answer - solution_vals.repeatability_solution10))

    images, coords, actions, rewards = gt.run_program(N=15)
    frames = utils_pset.raw_to_frame(images, coords, actions, rewards)
    student_answer = answer(frames, mapping)
    assert np.max(np.abs(student_answer - solution_vals.repeatability_solution15))

    images, coords, actions, rewards = gt.run_program(N=20)
    frames = utils_pset.raw_to_frame(images, coords, actions, rewards)
    student_answer = answer(frames, mapping)
    assert np.max(np.abs(student_answer - solution_vals.repeatability_solution20))

if __name__ == "__main__":
    f = open("solution_vals.py", "w")
    mapping = np.arange(200).reshape((2, 100))/200
    for i in range(5,25,5):
        sol = check_solution(solutions.temporal_cohesion_sol, solutions.temporal_cohesion_sol, mapping, i=i)
        f.write("temporal_solution{} = np.{}\n".format(i, repr(sol)))

    for i in range(5,25,5):
        sol = check_solution(solutions.causality_prior_sol, solutions.causality_prior_sol, mapping, i=i)
        f.write("causal_solution{} = np.{}\n".format(i, repr(sol)))

    for i in range(5,25,5):
        sol = check_solution(solutions.repeatability_prior_sol, solutions.repeatability_prior_sol, mapping, i=i)
        f.write("repeatability_solution{} = np.{}\n".format(i, repr(sol)))




