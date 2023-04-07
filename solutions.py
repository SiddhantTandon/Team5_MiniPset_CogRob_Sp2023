import numpy as np

class Data_Frame:

    def __init__(self, coords, action, reward, image):
        """
        Initialize a data frame
        :param coords: a numpy array of the coordinates of the robot
        :param action: a string representing the action taken
        :param reward: an integer representing the reward received
        :param image: a numpy array of the pixels of the image (flattened)
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

    time_steps = len(batch)
    total_loss_grad = np.zeros(mapping.shape)
    for i in range(0,time_steps-1):
        loss_grad = temporal_loss_gradient(batch[i], batch[i+1], mapping)
        total_loss_grad += loss_grad.reshape(2,4)

    return total_loss_grad/(time_steps-1)

def temporal_loss_gradient(init_state, next_state, mapping):
    dims = mapping.shape
    output = np.zeros(dims)
    for i in range(0, dims[0]):
        outside_comp = mapping[i,:]@init_state.image - mapping[i,:]@next_state.image
        for j in range(0, dims[1]):
            output[i,j] = 2*outside_comp*(init_state.image[j] - next_state.image[j])
    return output

image1 = np.array([0,1,1,1]).T #(0,0)
image2 = np.array([1,0,1,1]).T #(1,0)
image3 = np.array([1,1,0,1]).T #(0,0)
image4 = np.array([1,1,1,0.1]).T #(1,0)
frame1 = Data_Frame(np.array([1,2]), 4, 1, image1)
frame2 = Data_Frame(np.array([2,3]), 4, 2, image2)
frame3 = Data_Frame(np.array([2,1]), 4, 1, image3)
frame4 = Data_Frame(np.array([2,4]), 4, 2, image4)
mapping = np.arange(8).reshape((2,4))

print(temporal_cohesion_sol([frame1,frame2], mapping))


##TODO: write up the data set data structure as well as

def proportionality_prior_sol(batch):
    """
    computes the gradient proportionality prior from the batch
    :param batch: a list of Data frames where batch[i] is the data frame from time step i
    :return: Proportionality loss defined as average over:
    If the same action is taken at t1 and t2: (||(s_{t2+1} - s_{t2}|| - ||(s_{t1+1} - s_{t1}||)^2

    We want the gradient of this
    """

    #TODO: fill this out to find times with matching actions


def proportional_loss_gradient(s1, s2, s3, s4, mapping):
    dims = mapping.shape
    output = np.zeros(dims)
    delta1 = mapping@s2 - mapping@s1
    delta2 = mapping@s4 - mapping@s3
    outest = (np.linalg.norm(delta2) - np.linalg.norm(delta1))*2

    for i in range(0, dims[0]):

        outer_denom_1 = delta1[i]*np.linalg.norm(delta1)
        outer_denom_2 = delta2[i]*np.linalg.norm(delta2)
        for j in range(0, dims[1]):
            frac_1 = (delta1[i]**2)*(s1[j]- s2[j])/outer_denom_1
            frac_2 = (delta2[i]**2)*(s3[j] - s4[j])/outer_denom_2
            output[i,j] = -(frac_2 - frac_1)*outest

    return output

print(proportional_loss_gradient(frame1.image, frame2.image, frame3.image, frame4.image, mapping))

def causality_prior_sol(batch):
    """
    computes the gradient causality prior from the batch
    :param batch: a list of Data frames where batch[i] is the data frame from time step i
    :return: Causality loss defined as average over:
    If the same action is taken at t1 and t2, but different rewards are received:
    e^(-||(s_{t2} - s_{t1}||)

    We want the gradient of this
    """

    #TODO: fill this out to find times with matching actions

def causal_loss_gradient(s1,s2,mapping):
    dims = mapping.shape
    output = np.zeros(dims)

    delta = mapping@s1 - mapping@s2
    delta_norm = np.linalg.norm(delta)
    for i in range(0,dims[0]):
        denom = 2*delta[i]*delta_norm
        for j in range(0,dims[1]):
            numer = delta[i]*np.exp(-delta_norm) * (s1[j] - s2[j])*delta[i]*2
            output[i,j] = -numer/denom
    return output

print(causal_loss_gradient(frame1.image, frame3.image, mapping))


def repeatability_prior_sol(s1, s2, s3, s4, mapping):
    """
    TODO: FOR JAKE
    :param batch:
    :return:
    """
    dims = mapping.shape
    output = np.zeros(dims)

    delta1 = (mapping@s2 - mapping@s1)
    delta2 = (mapping@s4 - mapping@s3)
    base_priorish_loss = np.linalg.norm(delta2 - delta1)**2
    base_causal_loss = np.exp(-np.linalg.norm(mapping@s3 - mapping@s1))
    causal_losses = causal_loss_gradient(s1, s3, mapping)

    for i in range(0,dims[0]):
        outer = delta2[i] - delta1[i]
        for j in range(0,dims[1]):
            inner = s4[j] - s3[j] - s2[j] + s1[j]
            output[i,j] = 2*inner*outer*base_causal_loss + causal_losses[i,j]*base_priorish_loss
    return output

print(repeatability_prior_sol(frame1.image, frame2.image, frame3.image, frame4.image, mapping))
