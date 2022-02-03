import numpy as np
### these functions are used for the graphs

def get_xy_coords(q):
    x1 = lengths[0]*np.sin(q[0])
    x2 = lengths[1]*np.sin(q[1]) + x1
    x3 = lengths[2] * np.sin(q[2]) + x2
    y1 = -lengths[0]*np.cos(q[0]) 
    y2 = -lengths[1]*np.cos(q[1]) + y1 
    y3 = -lengths[2] * np.cos(q[2]) + y2
    x = (x1, x2, x3)
    y = (y1, y2, y3)
    return x, y 

def get_polar_coords(p):
	x, y = get_xy_coords(p)
	r1 = np.sqrt(x[0]*x[0] + y[0]*y[0])
	r2 = np.sqrt(x[1]*x[1] + y[1]*y[1])
	r3 = np.sqrt(x[2]*x[2] + y[2]*y[2])
	return (angle_mod(p[0]), angle_mod(p[1]), angle_mod(p[2])), (r1, r2, r3)

def get_xy_velocity(p):
	dx1 = -lengths[0] * np.cos(p[0])*p[3]
	dx2 = -lengths[1] * np.cos(p[1])*p[4] + dx1
	dx3 = -lengths[2] * np.cos(p[2])*p[5] + dx2
	dy1 = -lengths[0] * np.sin(p[0])*p[3]
	dy2 = -lengths[1] * np.sin(p[1])*p[4] + dy1 
	dy3 = -lengths[2] * np.sin(p[2])*p[5] + dy2
	return (dx1, dx2, dx3), (dy1, dy2, dy3)

def kinetic_energy(p, n):
    dx, dy = get_xy_velocity(p)
    ek = np.zeros(n)
    for i in range(0, n):
        ek[i] += (dx[i]**2 + dy[i]**2) / 2
    return ek

def potential_energy(p, n):
    x, y = get_xy_coords(p)
    ep = np.zeros(n)
    for i in range(0, n):
        ep[i] = g*(masses[0]*y[i]) + g*(masses[0]*sum(lengths[0:i+1]))
    return ep

def total_energy(p, n):
	return kinetic_energy(p, n) + potential_energy(p, n) 
    
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


