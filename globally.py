import numpy as np
import copy
from pendulums_functions import *
''' class general for every simulation'''
class Simulation:
    def __init__(self, h_step,  time, time_max, frameforsec, f_int, g, output):
        self.h_step     = h_step
        self.time          = time
        self.time_max   = time_max 
        self.frameforsec= frameforsec
        self.f_int      = f_int
        self.g          = g
        self.output     = output

    def increment_time(self):
        self.time += self.h_step



class n_Pendulum(Simulation):
    def __init__(self, h_step,  t, time_max, frameforsec, f_int, g, output, type_pend, lengths, masses, thetas, omegas):
            Simulation.__init__(self, h_step,  t, time_max, frameforsec, f_int, g, output)
            self.type_pend = type_pend
            self.lengths   = lengths
            self.masses    = masses
            self.thetas    = angle_mod(thetas)
            self.omegas    = omegas

    def running(self, anim_name):
        print(f"\n Running {anim_name} {self} pendulum propagated with {self.f_int.__name__}")

    def percentage(self):
        print(f"  {int(100*self.time/self.time_max)} % Processing  ", end="\r") 

    def __str__(self):
        n_pend_string = {1: "single", 2: "double", 3: "triple"}
        return n_pend_string[self.type_pend]
    def f_accel(self):
        f_n = { 1: f_single, 2: f_double, 3: f_triple}
        return f_n[self.type_pend]
    def fps_jump(self):
        return max(int((1/self.h_step)/self.frameforsec), 1)

    def get_fps(self):
        return int(min(self.frameforsec*self.time_max, self.time_max/self.h_step))
    def get_u(self):
        return np.concatenate((self.thetas, self.omegas), axis = None)
    def set_u(self, thetas, omegas):
        self.thetas = angle_mod(thetas)
        self.omegas = omegas
    def get_q(self):
        return self.thetas
    def get_p(self):
        return self.omegas
    def set_q(self, q):
        self.thetas = angle_mod(q)
    def set_p(self, p):
        self.omegas = p
    def set_masses(self, masses):
        self.masses = masses
    def set_lengths(self, lengths):
        self.lengths = lengths
    def set_g(self, g):
        self.g = g
    def get_xy_coords(self):
        x1 = self.lengths[0]*np.sin(self.thetas[0])
        x2 = self.lengths[1]*np.sin(self.thetas[1]) + x1
        x3 = self.lengths[2] * np.sin(self.thetas[2]) + x2
        y1 = -self.lengths[0]*np.cos(self.thetas[0]) 
        y2 = -self.lengths[1]*np.cos(self.thetas[1]) + y1 
        y3 = -self.lengths[2] * np.cos(self.thetas[2]) + y2
        x = (x1, x2, x3)
        y = (y1, y2, y3)
        return x, y 

    def get_polar_coords(self):
        x, y = n_Pendulum.get_xy_coords(self)
        r1 = self.lengths[0]
        r2 = np.sqrt(x[1]*x[1] + y[1]*y[1]) 
        r3 = np.sqrt(x[2]*x[2] + y[2]*y[2])
        return (self.thetas[0], self.thetas[1], self.thetas[2]), (r1, r2, r3)

    def get_xy_velocity(self):
        dx1 = -self.lengths[0] * np.cos(self.thetas[0])*self.omegas[0]
        dx2 = -self.lengths[1] * np.cos(self.thetas[1])*self.omegas[1] + dx1
        dx3 = -self.lengths[2] * np.cos(self.thetas[2])*self.omegas[2] + dx2
        dy1 = -self.lengths[0] * np.sin(self.thetas[0])*self.omegas[0]
        dy2 = -self.lengths[1] * np.sin(self.thetas[1])*self.omegas[1] + dy1 
        dy3 = -self.lengths[2] * np.sin(self.thetas[2])*self.omegas[2] + dy2
        return (dx1, dx2, dx3), (dy1, dy2, dy3)

    def kinetic_energy(self):
        dx, dy = n_Pendulum.get_xy_velocity(self)
        ek = np.zeros(self.type_pend)
        for i in range(0, self.type_pend):
            ek[i] += (dx[i]**2 + dy[i]**2) / 2
        return ek

    def potential_energy(self):
        x, y = n_Pendulum.get_xy_coords(self)
        ep = np.zeros(self.type_pend)
        for i in range(0, self.type_pend):
            ep[i] = self.g*(self.masses[i]*y[i]) + self.g*(self.masses[i]*sum(self.lengths[0:i+1]))
        return ep

    def total_energy(self):
        return n_Pendulum.kinetic_energy(self) + n_Pendulum.potential_energy(self) 
     
          

'''class Butterfly:
    def __init_(self, g, type_pend, n_type_pend):
    self.n_type_pend = n_type_pend
    '''

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
    for theta in thetas:
        theta = thetas[i]
        while (theta < 0):
            theta+=(2*np.pi)
            theta = theta % (2*np.pi)
        while( theta > np.pi):
            theta-=(2*np.pi)
        thetas[i] = theta
        i+=1
    return thetas

