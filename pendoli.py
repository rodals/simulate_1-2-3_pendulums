#!/usr/bin/python3
import numpy as np
from  time import perf_counter
import copy
#files
import f_print as pr
import utilities
import numerical_integration as ni
import pendulums_functions 
import graphs
import anime as an
from globally import *

def main():
    # default initial conditions
    # initial angle in grad 
    lengths  = np.array([1., 1., 1.])
    masses   = np.array([1., 1., 1.])
    grads0 = np.array([135, 135, 135])
    thetas0 = grads0 * 2 * np.pi /360
    omegas0_grad = np.zeros(3)
    omegas0  = omegas0_grad * 2 * np.pi / 360

    g = 9.81
    h_step = 0.001
    time_simulation = 10
    frameforsec = 30

    # default number of pendulum
    type_pend = 3
    # default method of integration
    f_int = ni.runge_kutta4

    ## for the butterfly effect
    perturb = 1e-4
    n_pends = 40

    # dictionary to simplify life for input n other things
    n_pend_string = {1: "single", 2: "double", 3: "triple"}
    dict_mode = { 1: "angles",2:"velocities", 3: "masses", 4: "lengths", 5: "gravity", 0: "nothing"}
    d_f_int = {1:ni.forward_euler, 2:ni.backward_euler, 3:ni.semi_implicit_euler, 4: ni.symplectic_euler, 5:ni.stormer_verlet,  6: ni.velocity_verlet, 7: ni.two_step_adams_bashforth, 8: ni.crank_nicolson, 9: ni.runge_kutta4,}
    dict_animation = { 1: an.animate_pendulum_simple,2: an.animate_pendulum_detailed, 3: an.the_butterfly_effect}

    ####### MENU #######
    n_i =  input('''
            Method of Numerical integration? - N for working pendulum \n
            1 Forward Euler - 1, 2, 3 \n
            2 Backward Euler - 1, 2, 3 \n
            3 Semi-Implicit Euler - 1 \n
            4 Symplectic Euler - 1, 2, 3 \n
            5 Stormer Verlet - 1, 2, 3  \n
            6 Velocity Verlet - 1 \n   
            7 Two-step Adams-Bashforth - 1, 2, 3 \n
            8 Crank Nicolson - 1, 2, 3 \n
           [9] Runge Kutta 4 - 1, 2, 3 \n''')

    if (n_i.isdigit()): f_int = d_f_int[int(n_i)]
    elif (n_i): bye()

    y_n = input(f"Running with Default configuration? [Y/n] \n   N pendulum = {type_pend} \n   time step = {h_step}s \n   theta_0 = {grads0}grad \n   l = {lengths}m \n   m = {masses}Kg \n   fps = {frameforsec}s**-1 \n   time simulation = {time_simulation}s \n   g = {g}m/s**2 \n").lower()
    if (y_n == "n"):
        type_pend = (input("n pendula to simulate? [1, 2, [3]]  "))
        if( type_pend in ["1", "2", "3"]):
            type_pend = int(type_pend) 
        elif (not type_pend): type_pend = 3
        else: exit("n pendola not supported atm.")

        for i in range(type_pend):
            print(f"Pend n. {i+1} :")
            lengths[i] = right_input(f" Length [{lengths[i]}] m:   ", float, lengths[i])

            masses[i]  = right_input(f" Mass [{masses[i]}] Kg:   ", float, masses[i])
            thetas0[i] = right_input(f" Initial theta [{grads0[i]}] Grad:    ", float, thetas0[i])*2*np.pi/360
            omegas0[i] = right_input(f" Initial omega [{omegas0_grad[i]}] Grad/s:   ", float, omegas0[i])*2*np.pi/360


        g = right_input(f"Gravity [{g}] m/s**2 :   ", float, g)
        h_step = right_input(f"Default time steps [{h_step}] s :   ", float, h_step)
        time_simulation = right_input(f"Time simulation [{time_simulation}] s :   ", float, time_simulation)
        frameforsec = right_input(f"Fps [{frameforsec}] :  ", float, frameforsec)
     
    elif y_n != "y" and y_n:
            print("Wrong input.")
            exit()

    mode = right_input('''
    Select Mode: \n
           [1] Simple animated pendulum \n
            2 Detailed animated pendulum \n
            3 The Butterfly Effect \n''', int, 1)

    if mode == 1 or  mode == 2: 
        fileoutput = f"{dict_animation[mode].__name__}_{n_pend_string[type_pend]}_{f_int.__name__}.mp4"
        fileoutput = right_input("Name output gif [enter to default]:   ", str, fileoutput)
        pend = n_Pendulum(h_step, 0., time_simulation, frameforsec, f_int, g, fileoutput, type_pend, lengths, masses, thetas0, omegas0)
        pr.running(dict_animation[mode].__name__, n_pend_string[type_pend], f_int.__name__)

    elif mode == 3: 
        n_mode =  right_input("What to perturb?\n  [1] Angles \n   2 Angular Velocities \n   3 First Mass \n   4 Lengths (visualization works only for same lengths pendulum) \n   5 Gravity \n   0 Nothing \n  ", float, 1)

        perturb = right_input(f"Module of perturbation? [{perturb} grad | {perturb} grad/s | {perturb} Kg | {perturb} m | {perturb} m/s^2]   ", float, perturb)
        n_pends = right_input(f"Number of pendulums simulated? [n_pends]", int, n_pends)
        fileoutput = f"{dict_animation[mode].__name__}_{n_pend_string[type_pend]}-perturb_{dict_mode[n_mode]}-{perturb}-{f_int.__name__}.mp4"
        fileoutput = right_input("Name output gif [enter to default]:   ", str, fileoutput)
        print(fileoutput)
        set_n_mode = { 1: n_Pendulum.set_q, 2: n_Pendulum.set_p, 3: n_Pendulum.set_masses, 4: n_Pendulum.set_lengths, 5: n_Pendulum.set_g}
        var_n_mode = { 1: thetas0, 2: omegas0, 3: masses, 4: lengths, 5: g}
        pend = [n_Pendulum(h_step, 0, time_simulation, frameforsec, f_int, g, fileoutput, type_pend, lengths, masses, thetas0, omegas0) for i in range(n_pends)]
        for i in range(n_pends):
            if (not n_mode == 3):
                set_n_mode[n_mode](pend[i], var_n_mode[n_mode] + perturb * i)
            else: 
                set_n_mode[n_mode](pend[i], np.array([masses[0] + perturb*i, masses[1], masses[2]]))

    else: bye()

    t_start = perf_counter()
    pend[0].running(dict_animation[mode].__name__)
    dict_animation[mode](pend)
    t_end = perf_counter()

    print(f"Time execution: {t_end - t_start: .4}")
    print(f"Output file: {fileoutput}")

if __name__ == '__main__':
    main()
