"""
Created on Sun Aug 20 15:41:46 2017

@author: alir3z4r
"""

import argparse
import numpy as np
from gridworld import grid_world as gw


parser = argparse.ArgumentParser()
parser.add_argument('--destination', type=int, nargs='+', default=[0,0,4,4],
                    metavar='d1x d1y d2x d2y',
                    help='Destination points; format is d1x d1y d2x d2y ...]')
parser.add_argument('--iters', type=int, default=10, 
                    help="Number of iterations for policy evaluation")                    
parser.add_argument('--Nx', type=int, default=4, 
                    help="Horizontal length of the grid")
parser.add_argument('--Ny', type=int, default=4, 
                    help="Vertical length of the grid")                    
parser.add_argument('--pvec', nargs='+', type=float,
                    default=[0.25,0.25,0.25,0.25],
                    help="Probability of moving to 4 directions (n-e-s-w)")
parser.add_argument('--start', type=int, nargs='+', default=[2,2], 
                    help='Starting point')
parser.add_argument('--thresh', type=float, default=0.1, 
                    help="Threshold in change of value function for terminating \
                    policy evaluation")                    



if __name__ == '__main__':
    args = parser.parse_args()
    assert np.sum(args.pvec) == 1, "The sum of probabilities should be one."
    assert len(args.start) == 2, "The number of start arguments must be exactly 2."
    start_point = [(args.start[0],args.start[1])]   
    dests = args.destination
    assert len(dests)%2 == 0, "The number of destination arguments must be even."
    destinations = [(dests[2*i],dests[2*i+1]) for i in range(int(len(dests)/2))]
    sg = gw(nx=args.Nx, ny=args.Ny, pvec=args.pvec, step_reward=-1)    
    sg.set_start(start_point)
    sg.set_destinations(destinations)
    values = sg.policy_evaluation(iter_max=args.iters, epsilon=args.thresh)
    print('State Value Function after {:d} iterations = \n{}'.format(args.iters,values))
