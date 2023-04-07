import numpy as np

class Data_Frame:

    def __init__(self, coords, action, reward, image):
        """
        Initialize a data frame
        :param coords: a numpy array of the coordinates of the robot
        :param action: a string representing the action taken
        :param reward: an integer representing the reward received
        """
        self.coords = coords
        self.action = action
        self.reward = reward
        self.image = image

def temporal_cohesion_sol(batch, mapping):
    """
    computes the gradient of the temporal cohesion loss on the given batch of state representation
    :param batch: a list of Data frames where batch[i] is the data frame from time step i
    :return: the gradient of the temporal cohesion loss
    """

    time_steps = batch.size()[0]
    total_loss_grad = np.zeros(mapping.size)
    for i in range(0,time_steps-1):
        loss_grad = temporal_loss_gradient(batch[i].coords, batch[i+1].coords, mapping)
        total_loss_grad += loss_grad

    return total_loss_grad/time_steps

def temporal_loss_gradient(init_state, next_state, mapping):
    dims = mapping.size
    output = np.zeros(dims)
    for i in range(0, dims[0]):
        outside_comp = mapping[i,:]*init_state.image - mapping[i,:]*next_state.image
        for j in range(0, dims[1]):
            output[i,j] = 2*outside_comp*(init_state.image[j] - next_state.image[j])
    return output


frame1 = Data_Frame(np.array([1,2]), "right", 0)
frame2 = Data_Frame(np.array([2,2]), "right", 0)



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