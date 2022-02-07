#!/usr/bin/python3
import numpy as np
from time import perf_counter
import os

#files
from nPendulum import nPendulum
import utilities as ut
import pendulum_animations as anime
import numerical_integration  as ni

def complete_menu():
    # default initial conditions
    lengths_def  = np.array([1., 1., 1.])
    masses_def   = np.array([1., 1., 1.])
    degs0_def = np.array([135, 135, 135])
    thetas0_def = degs0_def * 2 * np.pi /360
    omegas0_deg_def = np.zeros(3)
    omegas0_def  = omegas0_deg_def * 2 * np.pi / 360

    lengths  = np.array([])
    masses   = np.array([])
    thetas0 = np.array([])
    omegas0  = np.array([])

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
    dict_animation = { 1: anime.animate_pendulum_simple,2: anime.animate_pendulum_energy, 3: anime.animate_pendulum_detailed, 4: anime.animate_the_butterfly_effect}

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
   [9] Runge Kutta 4 - 1, 2, 3 \n
  ->''')

    if (n_i.isdigit()): f_int = d_f_int[int(n_i)]
    elif (n_i): ut.bye()

    y_n = input(f'''
Running with Default configuration? [[y]/n] \n   
    N pendulum = {type_pend} \n   
    time step = {h_step}s \n   
    theta_0 = {degs0_def}deg \n
    omega_0 = {omegas0_deg_def}deg \n
    lengths = {lengths_def}m \n
    masses = {masses_def}Kg \n
    fps = {frameforsec}s**-1 \n
    time simulation = {time_simulation}s \n
    g = {g}m/s**2 \n
   ->''').lower()
    if (y_n == "n"):
        type_pend = (input("n pendula to simulate? [1, 2, [3]]  \n ->"))
        if( type_pend in ["1", "2", "3"]):
            type_pend = int(type_pend) 
        elif (not type_pend): type_pend = 3
        else: exit("n pendola not supported atm.")

        for i in range(type_pend):
            print(f"\nPend n. {i+1} :")
            lengths = np.append(lengths, ut.right_input(f" Length [{lengths_def[i]}] m:   ", float, lengths_def[i]))
            masses = np.append(masses, ut.right_input(f" Mass [{masses_def[i]}] Kg:   ", float, masses_def[i]))
            thetas0 = np.append(thetas0, ut.right_input(f" Initial theta [{degs0_def[i]}] deg:    ", float, degs0_def[i])*2*np.pi/360)
            omegas0 = np.append(omegas0, ut.right_input(f" Initial omega [{omegas0_deg_def[i]}] deg/s:   ", float, omegas0_deg_def[i])*2*np.pi/360)


        g = ut.right_input(f"Gravity [{g}] m/s**2 :   ", float, g)
        h_step = ut.right_input(f"Default time steps [{h_step}] s :   ", float, h_step)
        time_simulation = ut.right_input(f"Time simulation [{time_simulation}] s :   ", float, time_simulation)
        frameforsec = ut.right_input(f"Fps [{frameforsec}] :  ", float, frameforsec)
     
    elif y_n != "y" and y_n:
            print("Wrong input.")
            exit()
    else: 
        thetas0 = thetas0_def
        omegas0 = omegas0_def
        masses = masses_def
        lengths = lengths_def

    mode = ut.right_input('''
Select Mode: \n
   [1] Simple animated pendulum \n
    2  Simple + Energy animated pendulum \n
    3  Detailed animated pendulum \n
    4  The Butterfly Effect \n
  ->''', int, 1)

    if mode == 1 or  mode == 2 or mode == 3: 
        fileoutput = f"./video/{dict_animation[mode].__name__}_{n_pend_string[type_pend]}_{f_int.__name__}.mp4"
        fileoutput = ut.right_input("Name output gif [enter to default]:   ", str, fileoutput)
        pend = nPendulum(h_step, 0., time_simulation, frameforsec, f_int, g, fileoutput, type_pend, lengths, masses, thetas0, omegas0)
        pend.running(dict_animation[mode].__name__)

    elif mode == 4: 
        n_mode =  ut.right_input("What to perturb?\n  [1] Angles \n   2 Angular Velocities \n   3 First Mass \n   4 Lengths  \n   5 Gravity \n   0 Nothing \n  ", float, 1)
        perturb_m = np.zeros(5)
        perturb_m[n_mode-1] = perturb
        perturb = ut.right_input(f"Module of perturbation? [{perturb} deg | {perturb} deg/s | {perturb} Kg | {perturb} m | {perturb} m/s^2]   ", float, perturb)
        n_pends = ut.right_input(f"Number of pendulums simulated? [n_pends]", int, n_pends)
        fileoutput = f"./video/{dict_animation[mode].__name__}_{n_pend_string[type_pend]}-perturb_{dict_mode[n_mode]}-{perturb}-{f_int.__name__}.mp4"
        fileoutput = ut.right_input("Name output gif [enter to default]:   ", str, fileoutput)
        set_n_mode = { 1: nPendulum.set_q, 2: nPendulum.set_omegas, 3: nPendulum.set_masses, 4: nPendulum.set_lengths, 5: nPendulum.set_g}
        perturb_masses =  np.zeros(type_pend)
        perturb_masses[0] += perturb_m[2]
        var_n_mode = { 1: thetas0, 2: omegas0, 3: masses, 4: lengths, 5: g}
        pend = [nPendulum(h_step, 0, time_simulation, frameforsec, f_int, g + perturb_m[4]*i, fileoutput, type_pend, lengths + perturb_m[3]*np.ones(type_pend)*i, masses + perturb_masses*i, thetas0 + perturb_m[0]*np.ones(type_pend)*i, omegas0 + perturb_m[1]*np.ones(type_pend)*i) for i in range(n_pends)]
        pend[0].running(dict_animation[mode].__name__)

    else: ut.bye()

    t_start = perf_counter()
    os.makedirs('./video', exist_ok = True)
    if mode == 4:
        dict_animation[mode](pend, perturb, dict_mode[n_mode])
    else: dict_animation[mode](pend)
    t_end = perf_counter()

    print(f"Time execution: {t_end - t_start: .4}")
    print(f"Output file: {fileoutput}")


def input_menu(input_file):
# default initial conditions
# initial angle in deg 
    i = 0
    perturb = 1e-6
    n_mode = 1
    with open(input_file, "r") as fh:
        for line in fh:
            if (not line.startswith("#")):
                if i == 0:     
                    n_i  = int(line) 
                elif i == 1:     
                    type_pend  = int(line)
                elif i == 2:
                    lengths = np.fromiter(map(float, line.split(" ")), dtype=np.float)        
                elif i == 3:
                    masses = np.fromiter(map(float, line.split(" ")), dtype=np.float)        
                elif i == 4:
                    degs0 = np.fromiter(map(float, line.split(" ")), dtype=np.float)        
                elif i == 5:
                    omegas0_deg = np.fromiter(map(float, line.split(" ")), dtype=np.float)        
                elif i == 6:
                    g  = float(line)
                elif i == 7:
                    h_step  = float(line)
                elif i == 8:
                    time_simulation  = float(line)
                elif i == 9:
                    frameforsec  = int(line)
                elif i == 10:
                    mode = int(line)
                elif i == 11:
                    fileoutput = line.strip()
                elif i == 12:
                    n_mode = int(line)
                elif i == 13:
                    perturb = float(line)
                elif i == 14:
                    n_pends = int(line)

                i+=1

    d_f_int = {1:ni.forward_euler, 2:ni.backward_euler, 3:ni.semi_implicit_euler, 4: ni.symplectic_euler, 5:ni.stormer_verlet,  6: ni.velocity_verlet, 7: ni.two_step_adams_bashforth, 8: ni.crank_nicolson, 9: ni.runge_kutta4,}
    f_int = d_f_int[n_i]
    thetas0 = degs0 * 2 * np.pi /360
    omegas0  = omegas0_deg * 2 * np.pi / 360

    ## for the butterfly effect
    perturb_m = np.zeros(5)
    perturb_m[n_mode-1] = perturb
    # dictionary to simplify life for input n other things
    n_pend_string = {1: "single", 2: "double", 3: "triple"}
    dict_mode = { 1: "angles",2:"velocities", 3: "masses", 4: "lengths", 5: "gravity", 0: "nothing"}
    dict_animation = { 1: anime.animate_pendulum_simple,2: anime.animate_pendulum_energy, 3: anime.animate_pendulum_detailed, 4: anime.animate_the_butterfly_effect}

    if mode == 1 or  mode == 2 or mode == 3: 
        pend = nPendulum(h_step, 0., time_simulation, frameforsec, f_int, g, fileoutput, type_pend, lengths, masses, thetas0, omegas0)
        pend.running(dict_animation[mode].__name__)
    elif mode == 4: 
        set_n_mode = { 1: nPendulum.set_q, 2: nPendulum.set_omegas, 3: nPendulum.set_masses, 4: nPendulum.set_lengths, 5: nPendulum.set_g}
        perturb_masses =  np.zeros(type_pend)
        perturb_masses[0] += perturb_m[2]
        var_n_mode = { 1: thetas0, 2: omegas0, 3: masses, 4: lengths, 5: g}
        pend = [nPendulum(h_step, 0, time_simulation, frameforsec, f_int, g + perturb_m[4]*i, fileoutput, type_pend, lengths + perturb_m[3]*np.ones(type_pend)*i, masses + perturb_masses*i, thetas0 + perturb_m[0]*np.ones(type_pend)*i, omegas0 + perturb_m[1]*np.ones(type_pend)*i) for i in range(n_pends)]
        pend[0].running(dict_animation[mode].__name__)

    else: ut.bye()

    t_start = perf_counter()
    if mode == 4:
        dict_animation[mode](pend, perturb, dict_mode[n_mode])
    else: dict_animation[mode](pend)
    t_end = perf_counter()

    print(f"Time execution: {t_end - t_start: .4}")
    print(f"Output file: {fileoutput}")

