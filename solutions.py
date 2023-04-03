import numpy as np

def temporal_cohesion_sol(batch):
    """
    computes the gradient of the temporal cohesion loss on the given batch of state representation
    :param batch: the batch of state representation where batch[i,j] represents the jth element of the state representation
    at time i
    :return: the gradient of the temporal cohesion loss
    """

    time_steps = batch.size()[0]
    total_loss_grad = 0
    for i in range(0,time_steps-1):
        loss_grad = temporal_loss_gradient(batch[i], batch[i+1])
        total_loss_grad += loss_grad

    return total_loss_grad/time_steps

def temporal_loss_gradient(init_state, next_state):
    state_diff = next_state - init_state
    return np.sum(state_diff)*2