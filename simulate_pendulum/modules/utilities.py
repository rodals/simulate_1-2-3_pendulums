import numpy as np

def bye():
    print("Non valid input. Bye")
    exit()

def right_input(write, type_wanted, default_value): 
    temp = input(write)
    if (not temp): return default_value
    if type_wanted == float:
        if (temp.lstrip('-').replace('.','',1).replace('e-','',1).replace('e','',1).isdigit()):  return float(temp)

    elif type_wanted == int:
        if (temp.lstrip('-').replace('e', '', 1).isdigit()): return int(temp)

    elif type_wanted == str: return str(temp)

    bye()

def angle_mod(thetas):
    i = 0
    count_max = 500
    for theta in thetas:
        theta = thetas[i]
        j = 0
        while (theta < 0):
            theta+=(2*np.pi)
            theta = theta % (2*np.pi)
        while( theta > np.pi and j < count_max):
            theta-=(2*np.pi)
            j+=1
        if (j == 500):
            exit("Overflow angle. Retry changing parameters")
        thetas[i] = theta
        i+=1
    return thetas

