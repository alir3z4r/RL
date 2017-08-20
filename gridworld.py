"""
Created on Sun Aug 20 14:01:46 2017

@author: alir3z4r
"""
import numpy as np

class grid_world:
    
    def __init__(self, nx, ny, pvec, step_reward=-1):
        """
        Class constructor
        
        attrs:
            nx: size of grid along x
            ny: size of grid along y
            pvec: a list of length 4; showing the probability that agent moves
                  north, east, south, and west, respectivly
            values: state values; initialized as init_values
            step_reward: the reward in taking each step; a negative number if
                        the goal is to get to the destination asap
        """
        self.nx = nx
        self.ny = ny
        self.pvec = pvec
        init_values = np.zeros([nx,ny])
        self.values = init_values
        self.step_reward = step_reward

    def set_destinations (self, destinations):
        """
        Sets the destinations in GridWorld
        The format is [(dist1_x,dist1_y),(dist2_x,dist2_y),...]
        """
        self.destinations = destinations

    def set_start (self, start):
        """
        Sets the start point; the format is similar to destination but with 
        only 1 point
        """
        self.start = start
        
    def reset_values (self):
        """
        Resets the values of the grid; is used if we want to renew the experiment
        """
        self.values = np.zeros([self.nx,self.ny])

    def one_step_bellman (self):
        """
        one-step Bellman update equation.
        """
        pvec = self.pvec
        values = self.values
        dests = self.destinations
        nx = self.nx
        ny = self.ny
        step_reward = self.step_reward
        new_values = np.zeros(shape=np.shape(values))
        
        def val_prev(values, x, y, nx, ny, direction):
            """
            returns the coordinates of new point after movement and retains the
            value of the previous state
            """
            val = values[x,y]
            if direction == 'n':
                x -= 1
            elif direction == 'e':
                y += 1
            elif direction == 's':
                x += 1
            elif direction == 'w':
                y -= 1
            if x in range(nx) and y in range(ny):
                val = values[x,y]
            return val
                
                
        for x in range(nx):
            for y in range(ny):
                if (x,y) not in dests:
                    for (p,direction) in zip(pvec,['n','e','s','w']):
                        new_values[x,y] = new_values[x,y] + p*(step_reward + 
                                        val_prev(values, x, y, nx, ny, direction))
        self.values = new_values
        return new_values
        
    def policy_evaluation (self, iter_max, epsilon):
        """
        Evaluates the policy specified by pvec
        
        Params:
            iter_max: maximum number of iterations for evaluating the policy
            epsilon: minimum change for continuing iterations (if the change 
                    in value function is less than epsilon, the loop terminates)
        Returns:
            The state values
            
        """
        delta_val = 1e3
        it = 0
        while it < iter_max and delta_val > epsilon:
            old_values = self.values
            new_values = self.one_step_bellman()
            delta_val = np.linalg.norm(new_values-old_values)
            it += 1
        return np.round(new_values, decimals=2)
                                        
        
                                        