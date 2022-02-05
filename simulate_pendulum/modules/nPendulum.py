import numpy as np
from Simulation import Simulation
import utilities as ut
import pendulum_functions as pf
import numerical_integration as ni

class nPendulum(Simulation):
    def __init__(self, h_step,  t, time_max, frameforsec, f_int, g, output, type_pend, lengths, masses, thetas, omegas):
            Simulation.__init__(self, h_step,  t, time_max, frameforsec, f_int, g, output)
            self.type_pend = type_pend
            self.lengths   = lengths
            self.masses    = masses
            self.thetas    = ut.angle_mod(thetas)
            self.omegas    = omegas
            self.p = nPendulum.set_p_from_q_omegas(self)
    def running(self, anim_name):
        print(f"\n Running {anim_name} {self} pendulum propagated with {self.f_int.__name__}")
    def percentage(self):
        print(f"  {int(100*self.time/self.time_max)} % Processing  ", end="\r") 
    def __str__(self):
        n_pend_string = {1: "single", 2: "double", 3: "triple"}
        return n_pend_string[self.type_pend]
    def f_accel(self):
        f_n = { 1: pf.f_single, 2: pf.f_double, 3: pf.f_triple}
        return f_n[self.type_pend]
    def fps_jump(self):
        return max(int((1/self.h_step)/self.frameforsec), 1)

    def get_fps(self):
        return int(min(self.frameforsec*self.time_max, self.time_max/self.h_step))
    def get_u(self):
        return np.concatenate((self.thetas, self.omegas), axis = None)
    def set_u(self, thetas, omegas):
        self.thetas = ut.angle_mod(thetas)
        self.omegas = omegas
    def get_q(self):
        return self.thetas
    def get_p(self):
        return self.p
    def get_omegas(self):
        return self.omegas
    def set_q(self, q):
        self.thetas = ut.angle_mod(q)
    def set_p(self, p):
        self.p = p
    def set_p_from_q_omegas(self):
            # if Hamiltonian
        if ( (self.f_int == ni.symplectic_euler) or (self.f_int == ni.stormer_verlet)):
            dict_p = {1: pf.Hamiltonian_single_p, 2: pf.Hamiltonian_double_p, 3: pf.Hamiltonian_triple_p}
            self.p = dict_p[self.type_pend](self.thetas, self.omegas, self.time, self.lengths, self.masses, self.g)
            return self.p
    def set_omegas_from_p_q(self):
            # if Hamiltonian
        if ( (self.f_int == ni.symplectic_euler) or (self.f_int == ni.stormer_verlet)):
            dict_p = {1: pf.Hamiltonian_single_omegas, 2: pf.Hamiltonian_double_omegas, 3: pf.Hamiltonian_triple_omegas}
            self.omegas = dict_p[self.type_pend](self.thetas, self.p, self.time, self.lengths, self.masses, self.g)
            return self.omegas

    def set_omegas(self, omegas):
        self.omegas = omegas 
    def set_masses(self, masses):
        self.masses = masses
    def set_lengths(self, lengths):
        self.lengths = lengths
    def set_g(self, g):
        self.g = g
    def get_xy_coords(self):
        x = np.zeros(self.type_pend)
        y = np.zeros(self.type_pend)
        temp_x = 0
        temp_y = 0
        for i in range(self.type_pend):
            x[i] = self.lengths[i]*np.sin(self.thetas[i]) + temp_x
            y[i] = -self.lengths[i]*np.cos(self.thetas[i]) + temp_y
            temp_x = x[i]
            temp_y = y[i]
        return x, y 
    def get_polar_coords(self):
        x, y = nPendulum.get_xy_coords(self)
        r = np.zeros(self.type_pend)
        for i in range(self.type_pend):
            r[i] = np.sqrt(x[i]*x[i] + y[i]*y[i])
        return self.thetas, r
    def get_xy_velocity(self):
        dx = np.zeros(self.type_pend)
        dy = np.zeros(self.type_pend)
        temp_x = 0
        temp_y = 0
        for i in range(self.type_pend):
            dx[i] = -self.lengths[i] * np.cos(self.thetas[i])*self.omegas[i] + temp_x
            dy[i] = -self.lengths[i] * np.sin(self.thetas[i])*self.omegas[i] + temp_y
            temp_x = dx[i]
            temp_y = dy[i]
        return dx, dy 
    def kinetic_energy(self):
        dx, dy = nPendulum.get_xy_velocity(self)
        ek = np.zeros(self.type_pend)
        for i in range(0, self.type_pend):
            ek[i] += self.masses[i] * (dx[i]**2 + dy[i]**2) / 2
        return ek
    def potential_energy(self):
        x, y = nPendulum.get_xy_coords(self)
        ep = np.zeros(self.type_pend)
        for i in range(0, self.type_pend):
            ep[i] = self.g*(self.masses[i]*y[i]) + self.g*(self.masses[i]*sum(self.lengths[0:i+1]))
        return ep
    def total_energy(self):
        return nPendulum.kinetic_energy(self) + nPendulum.potential_energy(self) 
