import numpy as np

class Data_Frame:

    def __init__(self, coords, action, reward):
        """
        Initialize a data frame
        :param coords: a numpy array of the coordinates of the robot
        :param action: a string representing the action taken
        :param reward: an integer representing the reward received
        """
        self.coords = coords
        self.action = action
        self.reward = reward

def temporal_cohesion_sol(batch):
    """
    computes the gradient of the temporal cohesion loss on the given batch of state representation
    :param batch: a list of Data frames where batch[i] is the data frame from time step i
    :return: the gradient of the temporal cohesion loss
    """

    time_steps = batch.size()[0]
    total_loss_grad = 0
    for i in range(0,time_steps-1):
        loss_grad = temporal_loss_gradient(batch[i].coords, batch[i+1].coords)
        total_loss_grad += loss_grad

    return total_loss_grad/time_steps

def temporal_loss_gradient(init_state, next_state):
    state_diff = next_state - init_state
    return np.sum(state_diff)*2



##TODO: write up the data set data structure as well as

def proportionality_prior_sol(batch):
    """
    computes the gradient proportionality prior from the batch
    :param batch: a list of Data frames where batch[i] is the data frame from time step i
    :return: Proportionality loss defined as average over:
    If the same action is taken at t1 and t2: (||(s_{t2+1} - s_{t2}|| - ||(s_{t1+1} - s_{t1}||)^2

    We want the gradient of this
    TODO: FOR ALLEGRA
    """


def causality_prior_sol(batch):
    """
    computes the gradient causality prior from the batch
    :param batch: a list of Data frames where batch[i] is the data frame from time step i
    :return: Causality loss defined as average over:
    If the same action is taken at t1 and t2, but different rewards are received:
    e^(-||(s_{t2} - s_{t1}||)

    We want the gradient of this
    TODO: FOR SIDDHANT
    """

def repeatability_prior_sol(batch):
    """
    TODO: FOR JAKE
    :param batch:
    :return:
    """