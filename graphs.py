import numpy as np
### these functions are used for the graphs

def xy_to_line(x, y, n, n_pend):
    list_line = np.zeros((n, n_pend, 2))
    for j in range(n):
        for i in range(n_pend):
            list_line[j][i][0] = x[j][i]
            list_line[j][i][1] = y[j][i]
    return list_line 

def xy_to_segment(x, y, n, n_pend):
    segments = np.zeros((n_pend, (n+1), 2))
    for i in range(n_pend):
        segments[i][0][0] = 0
        segments[i][0][1] = 0
        
    for j in range(n):
        for i in range(n_pend):
            segments[i][j+1][0] = x[j][i]
            segments[i][j+1][1] = y[j][i]
    return segments


