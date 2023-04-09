# Author: Siddhant Tandon
# Date: 4 April 2023

import numpy as np

class Data_Frame:

    def __init__(self, coords, action, reward, image):
        """
        Initialize a data frame
        :param coords: a numpy array of the coordinates of the robot
        :param action: an integer representing the action taken
        :param reward: an integer representing the reward received
        :param image: a numpy array of the pixels of the image (flattened)
        """
        self.coords = coords
        self.action = action
        self.reward = reward
        self.image = image


class Ground_truth():
    '''
    Class that runs the simulation and produces desired np.arrays for the following:
    1. flattened frame vector per simulation step
    2. Coordinate of the agent
    3. Action taken at this state
    4. Reward recieved to get to this state
    '''
    
    def run_program(N=25):
        '''
        Input: (optional)
        Number of simulation steps, N
        By default produces data for 25 steps
        Output:
        1. Array of flattened frame vectors per simulation step
        2. Array of Coordinate points of the agent
        3. Array of Action taken at each state
        4. Array of Reward recieved to get to the state
        
        '''
        
        # storing flat array
        image_flat = list()
        
        # storing agent coordinates
        coord = list()
        
        # storing agent actions
        action = list()
        
        # storing rewards
        rewards = list()
        
        # optional for matrix
        big = list()
        
        # setup 
        arr = np.ones(100, dtype=int) * 255
        mat = arr.reshape(10,10)
        x = [2]
        y = [2]
        i = 0
        mat[x[-1]][y[-1]] = 1
        reward = np.power((np.power(0 - x[-1], 2) + np.power(0-y[-1], 2)), 0.5)
        rewards.append(reward)
        image_flat.append(mat.reshape(100,1))
        tmp = np.array([x[-1], y[-1]])
        coord.append(tmp)
        
        # simulation loop
        while i < N:
            if y[-1] < 7 and x[-1] == 2:
                y.append( y[-1] + 1)
                action.append(1) # up
            elif x[-1] < 7 and y[-1] == 7:
                x.append(x[-1] + 1)
                action.append(4) # right
            elif x[-1] == 7 and y[-1] <= 7 and y[-1] > 2:
                y.append(y[-1] - 1)
                action.append(2) # down
            elif y[-1] == 2 and x[-1] <= 7 and x[-1] > 2:
                x.append(x[-1] - 1)
                action.append(3) # left
            
            reward = np.power((np.power(0 - x[-1], 2) + np.power(0-y[-1], 2)), 0.5)
            rewards.append(reward)
            arr = np.ones(100, dtype=int) * 255
            mat = arr.reshape(10,10)
            mat[x[-1]][y[-1]] = 1
            big.append(mat)
            image_flat.append(mat.reshape(100,1))
            tmp = np.array([x[-1], y[-1]])
            coord.append(tmp)        
            i += 1

        return np.array(image_flat), np.array(coord), np.array(action), np.array(rewards)
        