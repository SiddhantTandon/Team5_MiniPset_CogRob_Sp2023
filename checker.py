import utils_pset
import solutions
import numpy as np
import solution_vals

def temporal_sol_check(answer, threshold= 0.001):
    """
    Checks student answer for temporal loss
    :param answer: student temporal loss gradient function
    :return: None, b
    """
    mapping = np.arange(200).reshape((2, 100)) / 200
    gt = utils_pset.Ground_truth()

    images, coords, actions, rewards = gt.run_program(N=5)
    frames = utils_pset.raw_to_frame(images, coords, actions, rewards)
    student_answer = answer(frames, mapping)
    assert np.max(np.abs(student_answer - solution_vals.temporal_solution5)) < threshold

    images, coords, actions, rewards = gt.run_program(N=10)
    frames = utils_pset.raw_to_frame(images, coords, actions, rewards)
    student_answer = answer(frames, mapping)
    assert np.max(np.abs(student_answer - solution_vals.temporal_solution10)) < threshold

    images, coords, actions, rewards = gt.run_program(N=15)
    frames = utils_pset.raw_to_frame(images, coords, actions, rewards)
    student_answer = answer(frames, mapping)
    assert np.max(np.abs(student_answer - solution_vals.temporal_solution15)) < threshold

    images, coords, actions, rewards = gt.run_program(N=20)
    frames = utils_pset.raw_to_frame(images, coords, actions, rewards)
    student_answer = answer(frames, mapping)
    assert np.max(np.abs(student_answer - solution_vals.temporal_solution20)) < threshold

def causal_sol_check(answer, threshold = 0.001):
    mapping = np.arange(200).reshape((2, 100)) / 200
    gt = utils_pset.Ground_truth()

    images, coords, actions, rewards = gt.run_program(N=5)
    frames = utils_pset.raw_to_frame(images, coords, actions, rewards)
    student_answer = answer(frames, mapping)
    assert np.max(np.abs(student_answer - solution_vals.causal_solution5)) < threshold

    images, coords, actions, rewards = gt.run_program(N=10)
    frames = utils_pset.raw_to_frame(images, coords, actions, rewards)
    student_answer = answer(frames, mapping)
    assert np.max(np.abs(student_answer - solution_vals.causal_solution10)) < threshold

    images, coords, actions, rewards = gt.run_program(N=15)
    frames = utils_pset.raw_to_frame(images, coords, actions, rewards)
    student_answer = answer(frames, mapping)
    assert np.max(np.abs(student_answer - solution_vals.causal_solution15)) < threshold

    images, coords, actions, rewards = gt.run_program(N=20)
    frames = utils_pset.raw_to_frame(images, coords, actions, rewards)
    student_answer = answer(frames, mapping)
    assert np.max(np.abs(student_answer - solution_vals.causal_solution20)) < threshold


def repeatability_sol_check(answer, threshold = 0.001 ):
    mapping = np.arange(200).reshape((2, 100)) / 200
    gt = utils_pset.Ground_truth()

    images, coords, actions, rewards = gt.run_program(N=5)
    frames = utils_pset.raw_to_frame(images, coords, actions, rewards)
    student_answer = answer(frames, mapping)
    assert np.max(np.abs(student_answer - solution_vals.repeatability_solution5)) < threshold

    images, coords, actions, rewards = gt.run_program(N=10)
    frames = utils_pset.raw_to_frame(images, coords, actions, rewards)
    student_answer = answer(frames, mapping)
    assert np.max(np.abs(student_answer - solution_vals.repeatability_solution10)) < threshold

    images, coords, actions, rewards = gt.run_program(N=15)
    frames = utils_pset.raw_to_frame(images, coords, actions, rewards)
    student_answer = answer(frames, mapping)
    assert np.max(np.abs(student_answer - solution_vals.repeatability_solution15)) < threshold

    images, coords, actions, rewards = gt.run_program(N=20)
    frames = utils_pset.raw_to_frame(images, coords, actions, rewards)
    student_answer = answer(frames, mapping)
    assert np.max(np.abs(student_answer - solution_vals.repeatability_solution20)) < threshold



