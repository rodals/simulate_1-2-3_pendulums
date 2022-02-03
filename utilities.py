import numpy as np
import copy
#### usefull function 

# function to keep angle between -pi and pi
def angle_mod(theta):
    while (theta < 0):
        theta+=(2*np.pi)
    theta = theta % (2*np.pi)
    if theta > np.pi:
        theta-=(2*np.pi)
    return theta

# function to calculate functions implicitly
def implicit(f, q, p, h_inc,  s, p_or_q):
    diff = np.array([2*s, 2*s, 2*s]) 
    d = np.zeros(3)
    count = 0
    count_max = 500
    while( (np.abs(diff) > s).any() and count < count_max):
        if(p_or_q =="q"): d1 = f(q+d, p)*h_inc
        elif(p_or_q =="p"):  d1 = f(q, p-d)*h_inc
        diff = d1 -  d
        d = copy.deepcopy(d1)
        count+=1
    return d

