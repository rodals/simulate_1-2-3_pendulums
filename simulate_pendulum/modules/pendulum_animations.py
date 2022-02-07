import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import collections

import pendulum_functions as pf
import numerical_integration 

#### Types of animation  of the program

def animate_pendulum_simple(pend):
    fig = plt.figure(figsize = (16, 9))
    fig.suptitle(f'{pend.full_txt()}', fontsize=10)
    position = { 'motion':0}
    n = pend.type_pend
    pend_b = copy.deepcopy(pend)
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
    l_max = np.sum((pend.lengths) [0:n:1])
    axes_v[position['motion']].set(xlim=(-l_max*1.2, l_max*1.2), ylim=(-l_max*1.2, l_max*1.2))

    color_tails = {0: 'xkcd:lime', 1: 'xkcd:peach', 2: 'xkcd:sky blue'}
    marker_face = {0: 'xkcd:bright green', 1: 'xkcd:salmon', 2: 'xkcd:azure'}
    ax_pend_lines = [[], [], [], [], []]

   # different for so it is better from a visual point of view

    # first add the tail so it visualize behind
#    ax_pend_lines[1].append(axes_v[position["motion"]].plot([], [], 'o',color = color_tails[n-1],markersize = 12, markerfacecolor = marker_face[n-1],linestyle='', markevery=[-1], markeredgecolor = 'k', animated = True)[0])

    ax_pend_lines[1].append(axes_v[position["motion"]].plot([], [], 'o-',color = color_tails[n-1],markersize = 4, markerfacecolor = marker_face[n-1],linewidth=1, markevery=1, markeredgecolor = 'k', animated = True)[0])

# then add the lines from a mass to another
    for p in range(n):
        ax_pend_lines[0].append(axes_v[position["motion"]].plot([], [], color='k', linestyle='-', linewidth=2, animated = True)[0])    
# then add the points
    for p in range(n):
# tail and points
        ax_pend_lines[1].append(axes_v[position["motion"]].plot([], [], 'o-',color = color_tails[p],markersize = 12, markerfacecolor = marker_face[p],linewidth=2, markevery=1, markeredgecolor = 'k', animated = True)[0])


 
    anim_text = axes_v[position['motion']].text(0.02, 0.98, '', transform=axes_v[position['motion']].transAxes)
    time_text = axes_v[position['motion']].text(0.02, 0.94, '', transform=axes_v[position['motion']].transAxes)
    energy_text = axes_v[position['motion']].text(0.02, 0.90, '', transform=axes_v[position['motion']].transAxes)
    energy_diff_text = axes_v[position['motion']].text(0.02, 0.86, '', transform=axes_v[position['motion']].transAxes)

     
    def init():
        for p in range(len(ax_pend_lines[0])):
            ax_pend_lines[0][p].set_data([], [])
        for p in range(len(ax_pend_lines[1])):
            ax_pend_lines[1][p].set_data([], [])
        time_text.set_text('')
        energy_text.set_text('')
        energy_diff_text.set_text('')
        anim_text.set_text(f'{pend.f_int.__name__}')
        return ax_pend_lines[0]+ax_pend_lines[1] + [time_text, energy_text, anim_text] 
         

    def animate(i):
        tail = 1000
        for x in range(pend.fps_jump()):
            if (pend.f_int == numerical_integration.two_step_adams_bashforth):
                ((thetas, omegas), (thetas_b, omegas_b))  = pend.f_int(pend, pend_b)
                pend.set_u(thetas, omegas)
                pend_b.set_u(thetas_b, omegas_b)
                pend.increment_time()
                pend_b.increment_time()
            elif (pend.f_int == numerical_integration.symplectic_euler) or (pend.f_int == numerical_integration.stormer_verlet):
                thetas, p_i = pend.f_int(pend)
                pend.set_q(thetas)
                pend.set_p(p_i)
                pend.set_omegas_from_p_q()
                pend.increment_time()
            else:    
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
        ax_pend_lines[1][0].set_data(x_plot[n-1][max(0, i-tail):i+1], y_plot[n-1][max(0,i-tail):i+1])
        for j in range(1, n):
        # line from the i-1 mass to the i mass
            ax_pend_lines[0][j].set_data([x[j-1], x[j]], [y[j-1], y[j]])
            ax_pend_lines[1][j].set_data(x_plot[j-1][i], y_plot[j-1][i])

        ax_pend_lines[1][n].set_data(x_plot[n-1][i], y_plot[n-1][i])
    
        time_text.set_text('Time = %.1f' % (pend.time))
        energy_text.set_text('Total Energy = %.9f J' % en_tot[i])
        energy_diff_text.set_text(f'Error Energy = {((en_tot[i]-en_tot[0])/en_tot[0])*100: .5} %')
        pend.percentage()
        return (ax_pend_lines[0]+ ax_pend_lines[1]+ ax_pend_lines[2] + ax_pend_lines[3] + ax_pend_lines[4])

    
    anim = animation.FuncAnimation(fig, func = animate, init_func = init, interval=max(1000/pend.frameforsec, pend.h_step*1000), frames = pend.get_fps(), repeat = False, blit = True)
    anim.save(pend.output)
    print(" Done             ")
    plt.close('all')
    return anim

def animate_pendulum_energy(pend):
    fig = plt.figure(figsize = (16, 9))
    fig.suptitle(f'{pend.full_txt()}', fontsize=10)
    position = { 'motion':0, 'energy_tot': 1}
    n = pend.type_pend
    pend_b = copy.deepcopy(pend)
# graphs 
    axes_v = [ fig.add_subplot(2, 1, 1), fig.add_subplot(2, 1, 2)]
# variables to plot
    t_plot = []
    en_tot = []
    en_p = []

    
    x_plot = [ [], [], [] ]
    y_plot = [ [], [], [] ]
    r_plot = [ [], [], [] ]

    l_max = np.sum((pend.lengths) [0:n:1])
    axes_v[position['motion']].set_aspect('equal', adjustable='box')
    axes_v[position['motion']].axis('off')
    axes_v[position['motion']].set(xlim=(-l_max*1.2, l_max*1.2), ylim=(-l_max*1.2, l_max*1.2))
    axes_v[position['energy_tot']].set_title("Total Energy", pad=20)


    energy = []
    energy.append(axes_v[position["energy_tot"]].plot([], [], 'o-',label = r'$E_{tot}$',color = 'red',markersize = 2, markerfacecolor = 'red',linewidth=2, markevery=1, markeredgecolor = 'k', animated = True)[0])    
    axes_v[position['energy_tot']].ticklabel_format(useMathText=True)
    axes_v[position['energy_tot']].set_xlim(0, pend.time_max)
    axes_v[position['energy_tot']].set_ylabel(r"$Energy (J)$")
    axes_v[position['energy_tot']].yaxis.tick_right()
    axes_v[position['energy_tot']].yaxis.set_label_position("left")
    axes_v[position['energy_tot']].legend()
    color_tails = {0: 'xkcd:lime', 1: 'xkcd:peach', 2: 'xkcd:sky blue'}
    marker_face = {0: 'xkcd:bright green', 1: 'xkcd:salmon', 2: 'xkcd:azure'}
    ax_pend_lines = [[], [], [], [], []]
   # different for so it is better from a visual point of view

    # first add the tail so it visualize behind

    ax_pend_lines[1].append(axes_v[position["motion"]].plot([], [], 'o-',color = color_tails[n-1],markersize = 4, markerfacecolor = marker_face[n-1],linewidth=1, markevery=1, markeredgecolor = 'k', animated = True)[0])


# then add the lines from a mass to another
    for p in range(n):
        ax_pend_lines[0].append(axes_v[position["motion"]].plot([], [], color='k', linestyle='-', linewidth=2, animated = True)[0])    
# then add the points
    for p in range(n):
# tail and points
        ax_pend_lines[1].append(axes_v[position["motion"]].plot([], [], 'o-',color = color_tails[p],markersize = 12, markerfacecolor = marker_face[p],linewidth=2, markevery=1, markeredgecolor = 'k', animated = True)[0])


 
    anim_text = axes_v[position['motion']].text(0.02, 0.98, '', transform=axes_v[position['motion']].transAxes)
    time_text = axes_v[position['motion']].text(0.02, 0.94, '', transform=axes_v[position['motion']].transAxes)
    energy_text = axes_v[position['motion']].text(0.02, 0.90, '', transform=axes_v[position['motion']].transAxes)
    energy_diff_text = axes_v[position['motion']].text(0.02, 0.86, '', transform=axes_v[position['motion']].transAxes)

     
    def init():
        for p in range(len(ax_pend_lines[0])):
            ax_pend_lines[0][p].set_data([], [])
        for p in range(len(ax_pend_lines[1])):
            ax_pend_lines[1][p].set_data([], [])
        time_text.set_text('')
        energy_text.set_text('')
        energy_diff_text.set_text('')
        anim_text.set_text(f'{pend.f_int.__name__}')
        return ax_pend_lines[0]+ax_pend_lines[1] + [time_text, energy_text, anim_text] 
         

    def animate(i):
        tail = 1000
        for x in range(pend.fps_jump()):
            if (pend.f_int == numerical_integration.two_step_adams_bashforth):
                ((thetas, omegas), (thetas_b, omegas_b))  = pend.f_int(pend, pend_b)
                pend.set_u(thetas, omegas)
                pend_b.set_u(thetas_b, omegas_b)
                pend.increment_time()
                pend_b.increment_time()
            elif (pend.f_int == numerical_integration.symplectic_euler) or (pend.f_int == numerical_integration.stormer_verlet):
                thetas, p_i = pend.f_int(pend)
                pend.set_q(thetas)
                pend.set_p(p_i)
                pend.set_omegas_from_p_q()
                pend.increment_time()
            else:    
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
        ax_pend_lines[1][0].set_data(x_plot[n-1][max(0, i-tail):i+1], y_plot[n-1][max(0,i-tail):i+1])
        for j in range(1, n):
        # line from the i-1 mass to the i mass
            ax_pend_lines[0][j].set_data([x[j-1], x[j]], [y[j-1], y[j]])
            ax_pend_lines[1][j].set_data(x_plot[j-1][i], y_plot[j-1][i])

        ax_pend_lines[1][n].set_data(x_plot[n-1][i], y_plot[n-1][i])

        energy[0].set_data(t_plot[::-1], en_tot[::-1])
        axes_v[position['energy_tot']].relim()
        axes_v[position['energy_tot']].autoscale_view(True, True, True)

        time_text.set_text('Time = %.1f' % (pend.time))
        energy_text.set_text('Total Energy = %.9f J' % en_tot[i])
        energy_diff_text.set_text(f'Error Energy = {((en_tot[i]-en_tot[0])/en_tot[0])*100: .5} %')
        pend.percentage()
        return (ax_pend_lines[0]+ ax_pend_lines[1]+ ax_pend_lines[2] + ax_pend_lines[3] + ax_pend_lines[4])

    
    anim = animation.FuncAnimation(fig, func = animate, init_func = init, interval=max(1000/pend.frameforsec, pend.h_step*1000), frames = pend.get_fps(), repeat = False, blit = True)
    anim.save(pend.output)
    print(" Done             ")
    plt.close('all')
    return anim

def animate_pendulum_detailed(pend):
    t = 0
    fig = plt.figure(figsize = (12, 12) )
    fig.suptitle(f'{pend.full_txt()}', fontsize=10)
    position = {
            'motion':0, 'position_y':      1,    'complete_motion':2,
        'position_x':3,  'polar':          4, 'energy_tot':5,
        'phase':     6,'energy_points':    7, 'energy_k_p':8
        }

    n = pend.type_pend
    pend_b = copy.deepcopy(pend)

    f_n = { 1: pf.f_single, 2: pf.f_double, 3: pf.f_triple}

# graphs    
    projection_polar = {1: 'polar',  2: None, 3: '3d'}
    axes_v =[
            fig.add_subplot(3, 3, 1), fig.add_subplot(3, 3, 2), fig.add_subplot(3, 3, 3),
            fig.add_subplot(3, 3, 4),fig.add_subplot(3,3,5, projection = projection_polar[n]), fig.add_subplot(3,3,6),
            fig.add_subplot(3,3,7), fig.add_subplot(3, 3, 8), fig.add_subplot(3, 3, 9)
            ]

    tails = [10, 10, 10]

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


    l_max = np.sum((pend.lengths) [0:n:1])

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
    axes_v[position['position_x']].set_ylim(0, pend.time_max)

    axes_v[position['position_y']].sharey(axes_v[position['motion']])
    axes_v[position['position_y']].set_xlabel(r"$t (s)$")
    axes_v[position['position_y']].set_ylabel(r"$y (m)$")
    axes_v[position['position_y']].xaxis.tick_top()
    axes_v[position['position_y']].xaxis.set_label_position("top")
    axes_v[position['position_y']].set_xlim(0, pend.time_max)

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
    axes_v[position['energy_tot']].set_xlim(0, pend.time_max)
    axes_v[position['energy_tot']].set_ylabel(r"$Energy (J)$")
    axes_v[position['energy_tot']].yaxis.tick_right()
    axes_v[position['energy_tot']].yaxis.set_label_position("right")
    axes_v[position['energy_tot']].legend()

    axes_v[position['energy_k_p']].set_title(r"$E_k$,  $E_p$, $E_{tot}$")
    axes_v[position['energy_k_p']].set_xlim(0, pend.time_max)
    axes_v[position['energy_k_p']].set_ylabel(r"$Energy (J)$")
    axes_v[position['energy_k_p']].set_xlabel(r"$t (s)$")
    axes_v[position['energy_k_p']].yaxis.tick_right()
    axes_v[position['energy_k_p']].yaxis.set_label_position("right")
    axes_v[position['energy_k_p']].legend()

    energy_text = axes_v[position['motion']].text(0.02, 0.98, '', transform=axes_v[position['motion']].transAxes)
    energy_diff_text = axes_v[position['motion']].text(0.02, 0.92, '', transform=axes_v[position['motion']].transAxes)
    time_text = axes_v[position['motion']].text(0.02, 0.86, '', transform=axes_v[position['motion']].transAxes)
    anim_text = axes_v[position['motion']].text(0.02, 0.80, '', transform=axes_v[position['motion']].transAxes)

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
        anim_text.set_text(f'{pend.f_int.__name__}')
        return (ax_pend_lines[0]+ ax_pend_lines[1] +  pos_x + pos_y + phase +  energy+ [time_text, energy_text, anim_text])
         

    def animate(i):

        for x in range(pend.fps_jump()):
            if (pend.f_int == numerical_integration.two_step_adams_bashforth):
                ((thetas, omegas), (thetas_b, omegas_b))  = pend.f_int(pend, pend_b)
                pend.set_u(thetas, omegas)
                pend_b.set_u(thetas_b, omegas_b)
                pend.increment_time()
                pend_b.increment_time()
            elif (pend.f_int == numerical_integration.symplectic_euler) or (pend.f_int == numerical_integration.stormer_verlet):
                thetas, p_i = pend.f_int(pend)
                pend.set_q(thetas)
                pend.set_p(p_i)
                pend.set_omegas_from_p_q()
                pend.increment_time()
            else:    
                thetas, omegas = pend.f_int(pend)
                pend.set_u(thetas, omegas)
                pend.increment_time()


        x, y = pend.get_xy_coords()
        vx, vy = pend.get_xy_velocity()
        theta, r = pend.get_polar_coords()
        ene_tot = pend.total_energy() 
        omega = pend.get_omegas()

        for k in range(n):
            x_plot[k].append(x[k])
            y_plot[k].append(y[k])
            vx_plot[k].append(vx[k])
            vy_plot[k].append(vy[k])
            r_plot[k].append(r[k])
            theta_plot[k].append(theta[k])
            omega_plot[k].append(omega[k])
            ene_tot_plot[k].append(ene_tot[k])

        t_plot.append(pend.time)
        en_tot.append(np.sum(ene_tot))
        en_k.append(np.sum(pend.kinetic_energy()))
        en_p.append(np.sum(pend.potential_energy()))


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
            axes_v[position['polar']].set_rmax(pend.lengths*1.1)
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
       
    
        time_text.set_text('Time = %.1f' % (pend.time))
        energy_text.set_text('Total Energy = %.7f J' % en_tot[i])
        energy_diff_text.set_text(f'Error Energy = {((en_tot[i]-en_tot[0])/en_tot[0])*100: .5} %')
        pend.percentage()
        return (ax_pend_lines[0]+ ax_pend_lines[1] + pos_x + pos_y + phase + energy + [time_text, energy_text])

    
    anim = animation.FuncAnimation(fig, func = animate, init_func = init, interval=max(1000/pend.frameforsec, pend.h_step*1000), frames = pend.get_fps(), repeat = False, blit = True)
    anim.save(pend.output)
    print(" Done             ")
    plt.close('all')
    return anim

def animate_the_butterfly_effect(pends, perturb, perturb_mode):
    n_pend = len(pends)
    pend_def = pends[0]
    pends_b = copy.deepcopy(pends)
    n = pend_def.type_pend
    f_n = { 1: pf.f_single, 2: pf.f_double, 3: pf.f_triple}
    # mode 0 : thetas mode 1: omegas  mode 2: 
    track_segments_plot = np.zeros((n_pend, pend_def.get_fps(), 2))

    segments = np.zeros((n_pend, (n+1), 2))

    # for graphing
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle(f'{pend_def.full_txt()}\n {n_pend} perturbation: {perturb} of {perturb_mode}', fontsize=10)
    ax.axis('off')
    l_max = np.sum((pends[n_pend-1].lengths[0:n:1]))*1.2 
    ax.set(xlim=(-l_max, l_max), ylim=(-l_max, l_max))

    p_segments = np.zeros((n_pend, 0, 2))
    track_segments = np.zeros((n_pend, 0, 2))
    color_lines = plt.cm.rainbow(np.linspace(0, 1, n_pend))
    line_pends = collections.LineCollection(p_segments, color = 'black')
    line_track_pends = collections.LineCollection(track_segments, colors = color_lines)
    ax.add_collection(line_track_pends)
    ax.add_collection(line_pends)
    points, = plt.plot([], [],'ok', lw = '1')

    anim_text = ax.text(0.02, 0.98, '', transform=ax.transAxes)
    time_text = ax.text(0.02, 0.94, '', transform=ax.transAxes)
    perturb_text = ax.text(0.02, 0.94, '', transform=ax.transAxes)
    
    q_pend = np.empty((n_pend, n))
    x_pend = np.empty((n_pend, n))
    y_pend = np.empty((n_pend, n))

    def init():
        line_pends.set_segments(np.zeros((n_pend, 0, 2)))
        line_track_pends.set_segments(np.zeros((n_pend, 0, 2)))
        points.set_data([], [])
        time_text.set_text('')
        anim_text.set_text(f'{pend_def.f_int.__name__}')

        return line_pends, points, line_track_pends, time_text

    def animate(i):
        position = np.zeros((n, 2, n_pend))

        for x in range(pend_def.fps_jump()):
            for o in range(n_pend):
                if (pends[o].f_int == numerical_integration.two_step_adams_bashforth):
                    ((thetas, omegas), (thetas_b, omegas_b))  = pends[o].f_int(pends[o], pends_b[o])
                    pends[o].set_u(thetas, omegas)
                    pends_b[o].set_u(thetas_b, omegas_b)
                    pends[o].increment_time()
                    pends_b[o].increment_time()
                else:    
                    thetas, omegas = pends[o].f_int(pends[o])
                    pends[o].set_u(thetas, omegas)
                    pends[o].increment_time()

        for o in range(n_pend):
                q_pend[o] = pends[o].get_q()
                x_pend[o], y_pend[o]  = pends[o].get_xy_coords()
        track_segments_plot[0:n_pend:1][:,i][:,0] = x_pend[:,n-1] 
        track_segments_plot[0:n_pend:1][:,i][:,1] = y_pend[:,n-1]
        p_segments = xy_to_segment(x_pend, y_pend, n, n_pend)
        lines = xy_to_line(x_pend, y_pend, n, n_pend)
        line_pends.set_segments(p_segments)
        line_track_pends.set_segments(track_segments_plot[:, 0:i+1])
        time_text.set_text('Time = %.1f' % (pend_def.time))

        x_point, y_point = lines.reshape(-1, 2).T
        points.set_data(x_point, y_point)
        pend_def.percentage()
        return line_pends, points, line_track_pends, time_text

    anim = animation.FuncAnimation(fig, func = animate, init_func = init, interval=max(1000/pend_def.frameforsec, pend_def.h_step*1000), frames = pend_def.get_fps(), repeat = False, blit = True)
    anim.save(pend_def.output)
    print(" Done             ")
    plt.close('all')
    return anim

def xy_to_line(x, y, n, n_pend):
    list_line = np.zeros((n, n_pend, 2))
    for j in range(n):
        for i in range(n_pend):
            list_line[j][i][0] = x[i][j]
            list_line[j][i][1] = y[i][j]
    return list_line 

def xy_to_segment(x, y, n, n_pend):
    segments = np.zeros((n_pend, (n+1), 2))
    for i in range(n_pend):
        segments[i][0][0] = 0
        segments[i][0][1] = 0
        
    for j in range(n):
        for i in range(n_pend):
            segments[i][j+1][0] = x[i][j]
            segments[i][j+1][1] = y[i][j]
    return segments
