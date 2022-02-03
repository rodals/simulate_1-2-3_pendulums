import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import collections

from pendulums_functions import *

#### Types of animation  of the program

def animate_pendulum_simple(pend):
    fig = plt.figure(figsize = (16, 9))
    position = { 'motion':0}
    n = pend.type_pend
# graphs 
    axes_v = [ fig.add_subplot(1, 1, 1)]
# variables to plot
    t_plot = []
    en_tot = []
    en_p = []

    x_plot = [ [], [], [] ]
    y_plot = [ [], [], [] ]
    r_plot = [ [], [], [] ]

    axes_v[position['motion']].set_aspect('equal', adjustable='box')
    axes_v[position['motion']].axis('off')
    axes_v[position['motion']].set(xlim=(-(pend.lengths[0]+pend.lengths[1]+pend.lengths[2]), (pend.lengths[0]+pend.lengths[1]+pend.lengths[2])), ylim=(-(pend.lengths[0]+pend.lengths[1]+pend.lengths[2]), (pend.lengths[0]+pend.lengths[1]+pend.lengths[2])))

    color_tails = {0: 'xkcd:lime', 1: 'xkcd:peach', 2: 'xkcd:sky blue'}
    marker_face = {0: 'xkcd:bright green', 1: 'xkcd:salmon', 2: 'xkcd:azure'}
    ax_pend_lines = [[], [], [], [], []]

    for p in range(n):
# lines from a mass to another
        ax_pend_lines[0].append(axes_v[position["motion"]].plot([], [], color='k', linestyle='-', linewidth=2, animated = True)[0])    
    # different for so it is better from a visual point of view
    for p in range(n-1):
# tail and points
        ax_pend_lines[1].append(axes_v[position["motion"]].plot([], [], 'o-',color = color_tails[p],markersize = 12, markerfacecolor = marker_face[p],linewidth=2, markevery=[-1], markeredgecolor = 'k', animated = True)[0])

    ax_pend_lines[1].append(axes_v[position["motion"]].plot([], [], 'o-',color = color_tails[n-1],markersize = 4, markerfacecolor = marker_face[n-1],lw=1, markevery=1, markeredgecolor = 'k', animated = True)[0])
    ax_pend_lines[1].append(axes_v[position["motion"]].plot([], [], 'o',color = color_tails[n-1],markersize = 12, markerfacecolor = marker_face[n-1],linestyle='', markevery=[-1], markeredgecolor = 'k', animated = True)[0])

    time_text = axes_v[position['motion']].text(0.02, 0.95, '', transform=axes_v[position['motion']].transAxes)
    energy_text = axes_v[position['motion']].text(0.02, 0.90, '', transform=axes_v[position['motion']].transAxes)

     
    def init():
        for p in range(n):
            ax_pend_lines[0][p].set_data([], [])
        time_text.set_text('')
        energy_text.set_text('')
        return ax_pend_lines[0]+[time_text, energy_text] 
         

    def animate(i):
        tail = 100000
        for x in range(pend.fps_jump()):
            thetas, omegas = pend.f_int(pend)
            pend.set_u(thetas, omegas)
            pend.increment_time()

        x, y = pend.get_xy_coords()

        for k in range(n):
            x_plot[k].append(x[k])
            y_plot[k].append(y[k])

        t_plot.append(pend.time)
        ene_tot = pend.total_energy() 
        en_tot.append(np.sum(ene_tot))


        # line from the origin to the first mass
        ax_pend_lines[0][0].set_data([0, x[0]], [0, y[0]])
        for j in range(1, n):
        # line from the i-1 mass to the i mass
            ax_pend_lines[0][j].set_data([x[j-1], x[j]], [y[j-1], y[j]])
            ax_pend_lines[1][j-1].set_data(x_plot[j-1][i], y_plot[j-1][i])

        ax_pend_lines[1][n-1].set_data(x_plot[n-1][max(0, i-tail):i+1], y_plot[n-1][max(0,i-tail):i+1])
        ax_pend_lines[1][n].set_data(x_plot[n-1][i], y_plot[n-1][i])
    
        time_text.set_text('Time = %.1f' % (pend.time))
        energy_text.set_text('Total Energy = %.9f J' % en_tot[i])
        pend.percentage()
        return (ax_pend_lines[0]+ ax_pend_lines[1]+ ax_pend_lines[2] + ax_pend_lines[3] + ax_pend_lines[4])

    
    anim = animation.FuncAnimation(fig, func = animate, init_func = init, interval=max(1000/pend.frameforsec, pend.h_step*1000), frames = pend.get_fps(), repeat = False, blit = True)
    anim.save(pend.output)
    print(" Done             ")
    return anim


   


def animate_pendulum_detailed(pend):
    global t
    t = 0
    fig = plt.figure(figsize = (12, 12) )
    position = {
            'motion':0, 'position_y':      1,    'complete_motion':2,
        'position_x':3,  'polar':          4, 'energy_tot':5,
        'phase':     6,'energy_points':    7, 'energy_k_p':8
        }

    u0 = np.array([thetas0[0], thetas0[1], thetas0[2], omegas0[0], omegas0[1], omegas0[2]])
    um1 = np.array([thetas0[0], thetas0[1], thetas0[2], omegas0[0], omegas0[1], omegas0[2]])
    u = np.array([u0, um1])
    theta, r = get_polar_coords(u0)

    f_n = { 1: f_single, 2: f_double, 3: f_triple}

# graphs    
    projection_polar = {1: 'polar',  2: None, 3: '3d'}
    axes_v =[
            fig.add_subplot(3, 3, 1), fig.add_subplot(3, 3, 2), fig.add_subplot(3, 3, 3),
            fig.add_subplot(3, 3, 4),fig.add_subplot(3,3,5, projection = projection_polar[n]), fig.add_subplot(3,3,6),
            fig.add_subplot(3,3,7), fig.add_subplot(3, 3, 8), fig.add_subplot(3, 3, 9)
            ]

    tails = [10, 10, 10]
# how much should i jump writing frames to get at most frameforsec photos in a second
    fps_jump = max(int((1/h)/frameforsec), 1) 

# variables to plot
    t_plot = []
    en_tot = []
    en_tot_point = [[], [], []]
    en_k = []
    en_p = []
    x_plot = [ [], [], [] ]
    y_plot = [ [], [], [] ]
    vx_plot = [ [], [], [] ]
    vy_plot = [ [], [], [] ]
    r_plot = [ [], [], [] ]
    theta_plot = [ [], [], [] ]
    omega_plot = [ [], [], [] ]
    ene_tot_plot = [ [], [], [] ]

    energy = []
    pos_x = []
    pos_y = []
    phase = []


    color_tails = {0: 'xkcd:lime', 1: 'xkcd:peach', 2: 'xkcd:sky blue'}
    marker_face = {0: 'xkcd:bright green', 1: 'xkcd:salmon', 2: 'xkcd:azure'}
    ax_pend_lines = [[], [], [], [], []]
 
    for p in range(n):
# lines from a mass to another
        ax_pend_lines[0].append(axes_v[position["motion"]].plot([], [], color='k', linestyle='-', linewidth=2, animated = True)[0])    
    # draw only the last line of the pendulum
    complete_motion = axes_v[position["complete_motion"]].plot([], [], 'o-', color = color_tails[n-1],markersize = 4, markerfacecolor = marker_face[n-1], lw=2,  markevery=1, markeredgecolor = 'k', animated = True)[0]
    complete_motion_point = axes_v[position["complete_motion"]].plot([], [], 'o', color = color_tails[n-1],markersize = 12, markerfacecolor = marker_face[n-1], markevery=1, markeredgecolor = 'k', animated = True)[0]
# different for so it is better from a visual point of view
    for p in range(n):
# tail and points
        ax_pend_lines[1].append(axes_v[position["motion"]].plot([], [], 'o-',color = color_tails[p],markersize = 12, markerfacecolor = marker_face[p],linewidth=2, markevery=[0], markeredgecolor = 'k', animated = True)[0])

    energy.append(axes_v[position["energy_k_p"]].plot([], [], 'o-',label = r'$E_{k}$',color = 'blue',markersize = 4, markerfacecolor = 'blue',linewidth=2, markevery=[0], markeredgecolor = 'k', animated = True)[0])    
    energy.append(axes_v[position["energy_k_p"]].plot([], [], 'o-',label = r'$E_{p}$',color = 'orange',markersize = 4, markerfacecolor = 'orange', linewidth=2, markevery=[0], markeredgecolor = 'k', animated = True)[0])    
    energy.append(axes_v[position["energy_k_p"]].plot([], [], 'o-',label = r'$E_{tot}$',color = 'red',markersize = 4, markerfacecolor = 'red',linewidth=2, markevery=[0], markeredgecolor = 'k', animated = True)[0])    
    energy.append(axes_v[position["energy_tot"]].plot([], [], 'o-',label = r'$E_{tot}$',color = 'red',markersize = 2, markerfacecolor = 'red',linewidth=2, markevery=1, markeredgecolor = 'k', animated = True)[0])    

    for j in range(n):
        energy.append(axes_v[position["energy_points"]].plot([], [], 'o-',label = f'$E_{{{j+1}}}$',color = color_tails[j],markersize = 6, markerfacecolor = marker_face[j],linewidth=2, markevery=[0], markeredgecolor = 'k', animated = True)[0])    
        pos_x.append(axes_v[position['position_x']].plot([], [], 'o-', label = f'Posizione x{j+1}', color = color_tails[j], markersize = 6, markerfacecolor = marker_face[j], linewidth=2, markevery=[0], markeredgecolor = 'k', animated = True)[0])
        pos_y.append(axes_v[position['position_y']].plot([], [], 'o-',  label = f'Position y{j+1}', color = color_tails[j], markersize = 6, markerfacecolor = marker_face[j],linewidth=2, markevery=[0], markeredgecolor = 'k', animated = True)[0])
        string_theta = "\theta"
        string_dot_theta = "\dot{\theta}"
        phase.append(axes_v[position['phase']].plot([], [], 'o-', label = f'Phase ${string_theta}_{{{j+1}}}$ vs ${string_dot_theta}_{{{j+1}}}$', color = color_tails[j], markersize = 6, markerfacecolor = marker_face[j],linewidth=2, markevery=[0], markeredgecolor = 'k', animated = True)[0])


    l_max = np.sum((lengths) [0:n:1])

    axes_v[position['motion']].set_aspect('equal', adjustable='box')
    axes_v[position['motion']].axis('off')
    axes_v[position['motion']].set(xlim=(-l_max*1.2, l_max*1.2), ylim=(-l_max*1.2, l_max*1.2))

    axes_v[position['complete_motion']].set_title(r"Lissajous curve (last pendulum)")
    axes_v[position['complete_motion']].set_aspect('equal', adjustable='box')
    axes_v[position['complete_motion']].sharex(axes_v[position['motion']])
    axes_v[position['complete_motion']].sharey(axes_v[position['motion']])
    axes_v[position['complete_motion']].xaxis.tick_top()
    axes_v[position['complete_motion']].xaxis.set_label_position("top")

    axes_v[position['position_x']].sharex(axes_v[position['motion']])
    axes_v[position['position_x']].set_title(r"$x$ vs $t$")
    axes_v[position['position_x']].xaxis.tick_top()
    axes_v[position['position_x']].xaxis.set_label_position("top")
    axes_v[position['position_x']].set_xlabel(r"$x (m)$")
    axes_v[position['position_x']].set_ylabel(r"$t (s)$")
    axes_v[position['position_x']].set_ylim(0, time_simulation)

    axes_v[position['position_y']].sharey(axes_v[position['motion']])
    axes_v[position['position_y']].set_title(r"$t (s)$ vs $y (m)$")
    axes_v[position['position_y']].set_xlabel(r"$t (s)$")
    axes_v[position['position_y']].set_ylabel(r"$y (m)$")
    axes_v[position['position_y']].xaxis.tick_top()
    axes_v[position['position_y']].xaxis.set_label_position("top")
    axes_v[position['position_y']].set_xlim(0, time_simulation)

    axes_v[position['phase']].set_title(r"$\theta$ vs $\dot{\theta}$")
    axes_v[position['phase']].set_xlabel(r"$\theta (rad)$")
    axes_v[position['phase']].set_ylabel(r"$\dot{\theta} (rad/s)$")
    axes_v[position['phase']].set_xlim(-np.pi, np.pi)


    axes_v[position['energy_points']].set_ylabel(r"$Energy (J)$")
    axes_v[position['energy_points']].set_xlabel(r"$t (s)$")
    axes_v[position['energy_points']].yaxis.tick_right()
    axes_v[position['energy_points']].legend()

    axes_v[position['energy_tot']].set_title("Total Energy", pad=20)
    axes_v[position['energy_tot']].ticklabel_format(useMathText=True)
    axes_v[position['energy_tot']].set_xlim(0, time_simulation)
    axes_v[position['energy_tot']].set_ylabel(r"$Energy (J)$")
    axes_v[position['energy_tot']].yaxis.tick_right()
    axes_v[position['energy_tot']].yaxis.set_label_position("right")
    axes_v[position['energy_tot']].legend()

    axes_v[position['energy_k_p']].set_title(r"$E_k$,  $E_p$, $E_{tot}$")
    axes_v[position['energy_k_p']].set_xlim(0, time_simulation)
    axes_v[position['energy_k_p']].set_ylabel(r"$Energy (J)$")
    axes_v[position['energy_k_p']].set_xlabel(r"$t (s)$")
    axes_v[position['energy_k_p']].yaxis.tick_right()
    axes_v[position['energy_k_p']].yaxis.set_label_position("right")
    axes_v[position['energy_k_p']].legend()

    time_text = axes_v[position['motion']].text(0.02, 0.95, '', transform=axes_v[position['motion']].transAxes)
    energy_text = axes_v[position['motion']].text(0.02, 0.90, '', transform=axes_v[position['motion']].transAxes)

    def init():
        for p in range(n):
            ax_pend_lines[0][p].set_data([], [])
            ax_pend_lines[1][p].set_data([], [])
        complete_motion.set_data([], [])
        complete_motion_point.set_data([], [])

        for p in range(4):
            energy[p].set_data([], [])

        time_text.set_text('')
        energy_text.set_text('')
        return (ax_pend_lines[0]+ ax_pend_lines[1] +  pos_x + pos_y + phase +  energy+ [time_text, energy_text])
         

    def animate(i):
        global t
        nonlocal u
        for x in range(fps_jump):
            u = f(f_n[n], u)
            t+=h
        u0 = u[0]
        x, y = get_xy_coords(u0)
        vx, vy = get_xy_velocity(u0)
        theta, r = get_polar_coords(u0)
        ene_tot = total_energy(u0, n) 
        omega = u0[3:6]

        for k in range(n):
            x_plot[k].append(x[k])
            y_plot[k].append(y[k])
            vx_plot[k].append(vx[k])
            vy_plot[k].append(vy[k])
            r_plot[k].append(r[k])
            theta_plot[k].append(theta[k])
            omega_plot[k].append(omega[k])
            ene_tot_plot[k].append(ene_tot[k])

        t_plot.append(t)
        en_tot.append(np.sum(ene_tot))
        en_k.append(np.sum(kinetic_energy(u0, n)))
        en_p.append(np.sum(potential_energy(u0, n)))


# line from the origin to the first mass
        ax_pend_lines[0][0].set_data([0, x[0]], [0, y[0]])
        for j in range(1, n):
# line from the i-1 mass to the i mass
            ax_pend_lines[0][j].set_data([x[j-1], x[j]], [y[j-1], y[j]])

        for j in range(n):
            pos_x[j].set_data(x_plot[j][::-1], t_plot[::-1])
            pos_y[j].set_data(t_plot[::-1], y_plot[j][::-1])
            phase[j].set_data(theta_plot[j][::-1], omega_plot[j][::-1])
            ax_pend_lines[1][j].set_data(x_plot[j][i+1:max(1, i+1-tails[j]):-1], y_plot[j][i+1:max(1,i+1-tails[j]):-1])
        axes_v[position['polar']].clear()
        if (n == 3):
            axes_v[position['polar']].plot(theta_plot[0][::-1], theta_plot[1][::-1], theta_plot[2][::-1], 'o-',color = color_tails[2],markersize = 2, markerfacecolor = marker_face[0],linewidth=2, markevery=1, markeredgecolor = 'k', animated = True)
            axes_v[position['polar']].plot(theta_plot[0][i], theta_plot[1][i], theta_plot[2][i], 'o',color = 'xkcd:bright blue',markersize = 8, markerfacecolor = 'azure',linewidth=2, markevery=1, markeredgecolor = 'k', animated = True)
            axes_v[position['polar']].set_xlabel(r"$\theta_1 (rad)$")
            axes_v[position['polar']].set_ylabel(r"$\theta_2 (rad)$")
            axes_v[position['polar']].set_zlabel(r"$\theta_3 (rad)$")
        elif (n ==2 ):
            axes_v[position['polar']].plot(theta_plot[0][::-1], theta_plot[1][::-1], 'o-',color = 'xkcd:bright blue',markersize = 2, markerfacecolor = 'azure',linewidth=2, markevery=1, markeredgecolor = 'k', animated = True)
            axes_v[position['polar']].plot(theta_plot[0][i], theta_plot[1][i], 'o',color = 'xkcd:bright blue',markersize = 8, markerfacecolor = 'azure',ls='', markevery=1, markeredgecolor = 'k', animated = True)
            axes_v[position['polar']].set_xlabel(r"$\theta_1 (rad)$")
            axes_v[position['polar']].set_ylabel(r"$\theta_2 (rad)$")
            axes_v[position['polar']].yaxis.tick_right()
        elif (n == 1):
            axes_v[position['polar']].set_theta_zero_location("S")
            axes_v[position['polar']].set_rmax(lengths[0])
            axes_v[position['polar']].plot(theta_plot[0][::-1], r_plot[0][::-1], 'o-',color = 'xkcd:bright blue',markersize = 2, markerfacecolor = 'azure',linewidth=2, markevery=1, markeredgecolor = 'k', animated = True)
            axes_v[position['polar']].plot(theta_plot[0][i], r_plot[0][i], 'o',color = 'xkcd:bright blue',markersize = 8, markerfacecolor = 'azure',ls='', markevery=1, markeredgecolor = 'k', animated = True)


        complete_motion.set_data(x_plot[j], y_plot[j])
        complete_motion_point.set_data(x[n-1], y[n-1])

        energy[0].set_data(t_plot[::-1], en_k[::-1])
        energy[1].set_data(t_plot[::-1], en_p[::-1])
        energy[2].set_data(t_plot[::-1], en_tot[::-1])
        energy[3].set_data(t_plot[::-1], en_tot[::-1])
        for j in range(1, n + 1):
            energy[3+j].set_data(t_plot[::-1], ene_tot_plot[j-1][::-1])


        axes_v[position['phase']].relim()
        axes_v[position['position_x']].relim()
        axes_v[position['position_y']].relim()
        axes_v[position['energy_k_p']].relim()
        axes_v[position['energy_tot']].relim()
        axes_v[position['energy_points']].relim()

        axes_v[position['phase']].autoscale_view(True, True, True)
        axes_v[position['position_x']].autoscale_view(True, True, True)
        axes_v[position['position_y']].autoscale_view(True, True, True)
        axes_v[position['energy_k_p']].autoscale_view(True, True, True)
        axes_v[position['energy_tot']].autoscale_view(True, True, True)
        axes_v[position['energy_points']].autoscale_view(True, True, True)
       
    
        time_text.set_text('Time = %.1f' % (t))
        energy_text.set_text('Total Energy = %.7f J' % en_tot[i])
        percentage(i)
        return (ax_pend_lines[0]+ ax_pend_lines[1] + pos_x + pos_y + phase + energy + [time_text, energy_text])

    
    anim = animation.FuncAnimation(fig, func = animate, init_func = init, interval=max(1000/frameforsec, h*1000), frames = int(min(frameforsec * time_simulation, time_simulation/h)), repeat = False, blit = True)
    anim.save(output)
    print(" Done             ")
#    plt.show()
    return anim

def the_butterfly_effect(pends):
    global t #, lengths, masses
    t = 0

    # mode 0 : thetas mode 1: omegas  mode 2: 
    u0_pend = np.empty((n_pend, 6))
    um1_pend = np.empty((n_pend, 6))
    perturbation_th_omega = np.zeros(6)
    perturbation_masses = np.zeros(3)
    perturbation_lengths = np.zeros(3)
    perturbation_gravity = 0.
    frames = int(min(frameforsec * time_simulation, time_simulation/h))
    track_segments_plot = np.zeros((n_pend, frames, 2))

    if n_mode == 1:
        perturbation_th_omega[0:3] += perturbation
    elif n_mode == 2:
        perturbation_th_omega[3:6] += perturbation
    elif n_mode == 3:
        perturbation_masses =  np.array([perturbation*1 , 0, 0]) 
    elif n_mode == 4:
        perturbation_lengths += perturbation
    elif n_mode == 5:
        perturbation_gravity += perturbation
         
    u0 = np.array([thetas0[0], thetas0[1], thetas0[2], omegas0[0], omegas0[1], omegas0[2]])
    um1= np.array([thetas0[0], thetas0[1], thetas0[2], omegas0[0], omegas0[1], omegas0[2]])
    segments = np.zeros((n_pend, (n+1), 2))

    for i in range(n_pend):
        u0_pend[i] = u0 + perturbation_th_omega*i
        um1_pend[i] = u0 + perturbation_th_omega*i

# temporary fix for lengths TO CHANGE
    lengths_rapp = np.zeros(n_pend)
    for i in range(n_pend):
        lengths_rapp[i] = (lengths[0]+i*perturbation_lengths[0])/lengths[0]

    f_n = { 1: f_single, 2: f_double, 3: f_triple}

    # for graphing
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')
    l_max = np.sum((lengths*1.2 + n_pend*perturbation_lengths) [0:n:1])
    ax.set(xlim=(-l_max, l_max), ylim=(-l_max, l_max))

    p_segments = np.zeros((n_pend, 0, 2))
    track_segments = np.zeros((n_pend, 0, 2))
    color_lines = plt.cm.rainbow(np.linspace(0, 1, n_pend))
    pends = collections.LineCollection(p_segments, color = 'black')
    track_pends = collections.LineCollection(track_segments, colors = color_lines)
    ax.add_collection(track_pends)
    ax.add_collection(pends)
    points, = plt.plot([], [],'ok', lw = '1')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        pends.set_segments(np.zeros((n_pend, 0, 2)))
        track_pends.set_segments(np.zeros((n_pend, 0, 2)))
        points.set_data([], [])
        time_text.set_text('')

        return pends, points, track_pends, time_text

    def animate(i):
        global t, lengths, masses, g
        nonlocal u0_pend, um1_pend
        position = np.zeros((n, 2, n_pend))
        fps_jump = max(int((1/h)/frameforsec), 1) # how much should i jump writing frames to get at most frameforsec photos in a second

        for x in range(fps_jump):
            for o in range(n_pend):
                (u0_pend[o], um1_pend[o]) = f(f_n[n], [u0_pend[o], um1_pend[o]])
                lengths += perturbation_lengths
                masses  += perturbation_masses
                g       += perturbation_gravity

            t+=h
            masses  -= perturbation_masses*n_pend
            lengths -= perturbation_lengths*n_pend
            g       -= perturbation_gravity*n_pend

        x_pend, y_pend = get_xy_coords(u0_pend[:,:3].T)*lengths_rapp
        track_segments_plot[0:n_pend:1][:,i][:,0] = x_pend[n-1] 
        track_segments_plot[0:n_pend:1][:,i][:,1] = y_pend[n-1]
        p_segments = xy_to_segment(x_pend, y_pend, n, n_pend)
        lines = xy_to_line(x_pend, y_pend, n, n_pend)
        pends.set_segments(p_segments)
        track_pends.set_segments(track_segments_plot[:, 0:i+1])
        time_text.set_text('Time = %.1f' % (t))
        x_point, y_point = lines.reshape(-1, 2).T
        points.set_data(x_point, y_point)
        percentage(i)
        return pends, points, track_pends, time_text

    anim = animation.FuncAnimation(fig, func = animate, init_func = init, interval=max(1000/frameforsec, h*1000), frames = int(min(frameforsec * time_simulation, time_simulation/h)), repeat = False, blit = True)
    anim.save(output)
    print(" Done             ")
    return anim

