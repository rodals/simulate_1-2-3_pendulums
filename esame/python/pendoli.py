import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import copy
from  time import perf_counter
#from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib import collections

def stormer_verlet(f, u):
    u0 = u[0]
    um1 = u[1]
    if(t == 0):
        u0[0:3] += um1[3:6]*h + f(um1, 0)[3:6]*h*h/2
        u0[3:6] = (u0[0:3] - um1[0:3])/h
    # derivazione per ottenere y_1 errore O(h)
    u1 = np.zeros(6)
    u1[0:3] = 2*u0[0:3] - um1[0:3]  + f(u0, t)[3:6]*h*h
    u1[3:6] = (u1[0:3] - u0[0:3])/h
    u[1] = u0
    u[0] = u1
    return u


def velocity_verlet(f, u):
    s = 1e-14 # valore di soglia per risolvere iterativamente equazione autoconsistente
    diff = [2*s, 2*s, 2*s]
    u0 = u[0]
    y_1 = np.zeros(6)
    # "y_0" prima stima
    y_1[0:3] = u0[0:3] + u0[3:6]*h  + f(u0, t)[3:6]*h*h/2 
    # "y_0" prima stima
    y_1[3:6] = u0[3:6] + f(u0, t)[3:6]*h/2
    add = np.zeros(6)
    while (np.abs(diff[0]) > s or np.abs(diff[1]) > s or np.abs(diff[2]) > s):
        diff = add[3:6]
        add =  f(y_1 + add, t)*h/2
        diff -= add[3:6]
    y_1[3:6]+=add[3:6]
    u[0] = y_1
    return u

def runge_kutta4(f, u):
    u0 = u[0]
    k1 = f(u0, t)*h
    k2 = f(u0 + k1/2, t +h/2)*h
    k3 = f(u0 + k2/2 , t + h/2)*h
    k4 = f(u0 + k3 , t + h)*h
    u0 = (u0 + (k1 + 2*k2 + 2*k3 + k4)/6.0)
    u[0] = u0
    return u


def eulero_esplicito(f, u):
    u0 = u[0]
    u0 = (u0 + f(u0 , t)*h)
    u[0] = u0
    return u


def eulero_implicito(f, u):
    s = 1e-12
    u0 = u[0]
    c = np.zeros(6)
    diff = np.array([2*s, 2*s, 2*s, 2*s, 2*s, 2*s])
    p = 0
    while (np.abs(diff)>s).any():
        diff = c
        c =  f(u0 + c, t+h)*h
        diff -= c
        p+=1
    u[0] += c        
    return u
            
def trapezoide_implicito(f, u):
    s = 1e-12
    u0 = u[0]
    c = np.zeros(6)
    diff = np.array([2*s, 2*s, 2*s, 2*s, 2*s, 2*s])
    p = 0
    u[0] += f(u0, t)*h/2

    while (np.abs(diff)>s).any():
        diff = c
        c =  f(u0 + c, t+h)*h/2
        diff -= c
        p+=1
    u[0] += c        
    return u
 
 
def eulero_semi_implicito(f, u):
    u0 = u[0]
    u0[3:6] += f(u0 , t)[3:6]*h
    u0[0:3] += f(u0 , t)[0:3]*h
    u[0] = u0
    return u

def f_triple(u, t):
	theta_1, theta_2, theta_3 = u[0], u[1], u[2]
	w1, w2, w3 = u[3], u[4], u[5]

	theta_21 = theta_2 - theta_1
	theta_31 = theta_3 - theta_1
	theta_32 = theta_3 - theta_2

	cos_21, cos_31, cos_32 = (np.cos(theta_21), np.cos(theta_31), np.cos(theta_32))
	sin_21, sin_31, sin_32 = (np.sin(theta_21), np.sin(theta_31), np.sin(theta_32))
	sin_1, sin_2, sin_3 = (np.sin(theta_1), np.sin(theta_2), np.sin(theta_3))
	cos_1, cos_2, cos_3 = (np.cos(theta_1), np.cos(theta_2), np.cos(theta_3))

	dw1 = (masses[2]*(g*sin_3 + lengths[0]*w1**2*(cos_1*sin_3 - cos_3*sin_1) + lengths[1]*w2**2*(cos_2*sin_3 - cos_3*sin_2))*(cos_21*cos_32*masses[1] + cos_21*cos_32*masses[2] - cos_31*masses[1] - cos_31*masses[2]) - (cos_21*masses[1] + cos_21*masses[2] - cos_31*cos_32*masses[2])*(g*masses[1]*sin_2 + g*masses[2]*sin_2 + lengths[0]*masses[1]*w1**2*(cos_1*sin_2 - cos_2*sin_1) + lengths[0]*masses[2]*w1**2*(cos_1*sin_2 - cos_2*sin_1) - lengths[2]*masses[2]*w3**2*(cos_2*sin_3 - cos_3*sin_2)) + (-cos_32**2*masses[2] + masses[1] + masses[2])*(g*masses[0]*sin_1 + g*masses[1]*sin_1 + g*masses[2]*sin_1 - lengths[1]*masses[1]*w2**2*(cos_1*sin_2 - cos_2*sin_1) - lengths[1]*masses[2]*w2**2*(cos_1*sin_2 - cos_2*sin_1) - lengths[2]*masses[2]*w3**2*(cos_1*sin_3 - cos_3*sin_1)))/(lengths[0]*(cos_21**2*masses[1]**2 + 2*cos_21**2*masses[1]*masses[2] + cos_21**2*masses[2]**2 - 2*cos_21*cos_31*cos_32*masses[1]*masses[2] - 2*cos_21*cos_31*cos_32*masses[2]**2 + cos_31**2*masses[1]*masses[2] + cos_31**2*masses[2]**2 + cos_32**2*masses[0]*masses[2] + cos_32**2*masses[1]*masses[2] + cos_32**2*masses[2]**2 - masses[0]*masses[1] - masses[0]*masses[2] - masses[1]**2 - 2*masses[1]*masses[2] - masses[2]**2))

	dw2 = (-masses[2]*(g*sin_3 + lengths[0]*w1**2*(cos_1*sin_3 - cos_3*sin_1) + lengths[1]*w2**2*(cos_2*sin_3 - cos_3*sin_2))*(-cos_21*cos_31*masses[1] - cos_21*cos_31*masses[2] + cos_32*masses[0] + cos_32*masses[1] + cos_32*masses[2]) - (cos_21*masses[1] + cos_21*masses[2] - cos_31*cos_32*masses[2])*(g*masses[0]*sin_1 + g*masses[1]*sin_1 + g*masses[2]*sin_1 - lengths[1]*masses[1]*w2**2*(cos_1*sin_2 - cos_2*sin_1) - lengths[1]*masses[2]*w2**2*(cos_1*sin_2 - cos_2*sin_1) - lengths[2]*masses[2]*w3**2*(cos_1*sin_3 - cos_3*sin_1)) + (-cos_31**2*masses[2] + masses[0] + masses[1] + masses[2])*(g*masses[1]*sin_2 + g*masses[2]*sin_2 + lengths[0]*masses[1]*w1**2*(cos_1*sin_2 - cos_2*sin_1) + lengths[0]*masses[2]*w1**2*(cos_1*sin_2 - cos_2*sin_1) - lengths[2]*masses[2]*w3**2*(cos_2*sin_3 - cos_3*sin_2)))/(lengths[1]*(cos_21**2*masses[1]**2 + 2*cos_21**2*masses[1]*masses[2] + cos_21**2*masses[2]**2 - 2*cos_21*cos_31*cos_32*masses[1]*masses[2] - 2*cos_21*cos_31*cos_32*masses[2]**2 + cos_31**2*masses[1]*masses[2] + cos_31**2*masses[2]**2 + cos_32**2*masses[0]*masses[2] + cos_32**2*masses[1]*masses[2] + cos_32**2*masses[2]**2 - masses[0]*masses[1] - masses[0]*masses[2] - masses[1]**2 - 2*masses[1]*masses[2] - masses[2]**2))

	dw3 = ((g*sin_3 + lengths[0]*w1**2*(cos_1*sin_3 - cos_3*sin_1) + lengths[1]*w2**2*(cos_2*sin_3 - cos_3*sin_2))*(-cos_21**2*masses[1]**2 - 2*cos_21**2*masses[1]*masses[2] - cos_21**2*masses[2]**2 + masses[0]*masses[1] + masses[0]*masses[2] + masses[1]**2 + 2*masses[1]*masses[2] + masses[2]**2) + (cos_21*cos_32*masses[1] + cos_21*cos_32*masses[2] - cos_31*masses[1] - cos_31*masses[2])*(g*masses[0]*sin_1 + g*masses[1]*sin_1 + g*masses[2]*sin_1 - lengths[1]*masses[1]*w2**2*(cos_1*sin_2 - cos_2*sin_1) - lengths[1]*masses[2]*w2**2*(cos_1*sin_2 - cos_2*sin_1) - lengths[2]*masses[2]*w3**2*(cos_1*sin_3 - cos_3*sin_1)) - (-cos_21*cos_31*masses[1] - cos_21*cos_31*masses[2] + cos_32*masses[0] + cos_32*masses[1] + cos_32*masses[2])*(g*masses[1]*sin_2 + g*masses[2]*sin_2 + lengths[0]*masses[1]*w1**2*(cos_1*sin_2 - cos_2*sin_1) + lengths[0]*masses[2]*w1**2*(cos_1*sin_2 - cos_2*sin_1) - lengths[2]*masses[2]*w3**2*(cos_2*sin_3 - cos_3*sin_2)))/(lengths[2]*(cos_21**2*masses[1]**2 + 2*cos_21**2*masses[1]*masses[2] + cos_21**2*masses[2]**2 - 2*cos_21*cos_31*cos_32*masses[1]*masses[2] - 2*cos_21*cos_31*cos_32*masses[2]**2 + cos_31**2*masses[1]*masses[2] + cos_31**2*masses[2]**2 + cos_32**2*masses[0]*masses[2] + cos_32**2*masses[1]*masses[2] + cos_32**2*masses[2]**2 - masses[0]*masses[1] - masses[0]*masses[2] - masses[1]**2 - 2*masses[1]*masses[2] - masses[2]**2))

	return np.array([ w1, w2, w3 , dw1, dw2, dw3])

def f_double( u , t):
    theta_1, theta_2 = u[0], u[1]
    w1, w2 = u[3], u[4]

    theta_12 = theta_1 - theta_2
    sin_1, sin_2 = (np.sin(theta_1), np.sin(theta_2))
    cos_1, cos_2 = (np.cos(theta_1), np.cos(theta_2))
    cos_12 = np.cos(theta_12)
    sin_12 = np.sin(theta_12)
    cos_21 = cos_12
    sin_21 = - sin_12

    dw1 = (-sin_12*(masses[1]*lengths[0]*w1**2 * cos_12 + masses[1]*lengths[1]*w2**2) - g*((masses[0]+masses[1])*sin_1 - masses[1]*sin_2 * cos_12))/(lengths[0]*(masses[0] + masses[1]* sin_12**2))
    dw2 = (sin_12 * ((masses[0]+masses[1])*lengths[0]*w1**2 + masses[1]*lengths[1]*w2**2 * cos_12) + g*((masses[0]+masses[1])*sin_1 * cos_12 - (masses[0]+masses[1])*sin_2))/(lengths[1]*(masses[0]+masses[1]*sin_12**2))   

    return np.array([ w1, w2, 0, dw1, dw2, 0])



def f_single( u , t):
	theta_1 = u[0]
	w1 = u[3]      
	dw1 = -g/lengths[0] * np.sin(theta_1)
	return np.array([ w1, 0, 0, dw1, 0, 0])



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
	return (p[0], p[1], p[2]), (r1, r2, r3)


def get_xy_velocity(p):
	dx1 = -lengths[0] * np.cos(p[0])*p[3]
	dx2 = -lengths[1] * np.cos(p[1])*p[4] + dx1
	dx3 = -lengths[2] * np.cos(p[2])*p[5] + dx2
	dy1 = -lengths[0] * np.sin(p[0])*p[3]
	dy2 = -lengths[1] * np.sin(p[1])*p[4] + dy1 
	dy3 = -lengths[2] * np.sin(p[2])*p[5] + dy2
	return (dx1, dx2, dx3), (dy1, dy2, dy3)

def energia_cinetica(p, n):
    dx, dy = get_xy_velocity(p)
    ek = 0
    for i in range(0, n):
        ek += (dx[i]**2 + dy[i]**2) / 2
    return ek


def energia_potenziale(p, n):
    x, y = get_xy_coords(p)
    ep = 0
    for i in range(0, n):
        ep += g*(masses[0]*y[i]) + g*(masses[0]*sum(lengths[0:i+1]))
    return ep

def energia_totale(p, n):
	return energia_cinetica(p, n) + energia_potenziale(p, n) 
    
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


def butterfly_effect(f, output, n, n_pend, perturbation, n_mode):
    global t, lengths, masses
    t = 0
    # mode 0 : thetas mode 1: omegas  mode 2: 
    u0_pend = np.empty((n_pend, 6))
    um1_pend = np.empty((n_pend, 6))
    perturbation_th_omega = np.zeros(6)
    perturbation_masses = np.zeros(3)
    perturbation_lengths = np.zeros(3)

    if n_mode == 1:
        perturbation_th_omega[0:3] += perturbation
    elif n_mode == 2:
        perturbation_th_omega[3:6] += perturbation
    elif n_mode == 3:
        perturbation_masses =  np.array([perturbation*1 , 0, 0]) 
    elif n_mode == 4:
        perturbation_lengths += perturbation
         
    u0 = np.array([thetas0[0], thetas0[1], thetas0[2], omegas0[0], omegas0[1], omegas0[2]])
    um1 = np.array([thetas0[0], thetas0[1], thetas0[2], omegas0[0], omegas0[1], omegas0[2]])
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

    points, = plt.plot([], [],'ok', lw = '1')
    p_segments = np.zeros((n_pend, 0, 2))
    color_lines = plt.cm.rainbow(np.linspace(0, 1, n_pend))
    pends = collections.LineCollection(p_segments, color = color_lines)
    ax.add_collection(pends)

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        pends.set_segments(np.zeros((n_pend, 0, 2)))
        points.set_data([], [])
        time_text.set_text('')
        return pends, points, time_text

    def animate(i):
        global t, lengths, masses
        nonlocal u0_pend, um1_pend
        position = np.zeros((n, 2, n_pend))
        fps_jump = max(int((1/h)/framepersec), 1) # ogni quanto devo saltare di scrivere i frame per ottenere al piu' framepersec  foto in un secondo
        for x in range(fps_jump):
            for o in range(n_pend):
                (u0_pend[o], um1_pend[o]) = f(f_n[n], [u0_pend[o], um1_pend[o]])
                lengths += perturbation_lengths
                masses  += perturbation_masses
            t+=h
            masses  -= perturbation_masses*n_pend
            lengths -= perturbation_lengths*n_pend

        x_pend, y_pend = get_xy_coords(u0_pend[:,:3].T)*lengths_rapp
        p_segments = xy_to_segment(x_pend, y_pend, n, n_pend)
        lines = xy_to_line(x_pend, y_pend, n, n_pend)
        pends.set_segments(p_segments)
        time_text.set_text('time = %.1f' % (t))
        
        x_point, y_point = lines.reshape(-1, 2).T
        points.set_data(x_point, y_point)
        print(f"  {int(100*i/min(framepersec * tempo_simulazione, int(tempo_simulazione/h)))} % Processing", end="\r") 
        return pends, points, time_text

    anim = animation.FuncAnimation(fig, func = animate, init_func = init, interval=max(1000/framepersec, h*1000), frames = int(min(framepersec * tempo_simulazione, tempo_simulazione/h)), repeat = False, blit = True)
    anim.save(output)
    print(" Done             ")
    #    plt.show()
    return anim


   


def animate_pendulum(f, output, n):
    global t
    t = 0
    fig = plt.figure()
    position = { 'motion':0, 'polar':1, 'energy_tot':2, 'energy_k_p':3, 'position': 4 }

    u0 = np.array([thetas0[0], thetas0[1], thetas0[2], omegas0[0], omegas0[1], omegas0[2]])
    um1 = np.array([thetas0[0], thetas0[1], thetas0[2], omegas0[0], omegas0[1], omegas0[2]])
    u = np.array([u0, um1])
    theta, r = get_polar_coords(u0)

    f_n = { 1: f_single, 2: f_double, 3: f_triple}

# graphs 
    axes_v = [fig.add_subplot(3, 3, 3,projection = 'polar'), fig.add_subplot(2, 2, 1),fig.add_subplot(3, 3, 6),fig.add_subplot(3, 3, 7)]
# variables to plot
    t_plot = []
    en_tot = []
    en_k = []
    en_p = []
    x_plot = [ [], [], [] ]
    y_plot = [ [], [], [] ]
    r_plot = [ [], [], [] ]
    theta_plot = [ [], [], [] ]

    axes_v[position['motion']].set_aspect('equal', adjustable='box')
    axes_v[position['motion']].axis('off')
    axes_v[position['motion']].set(xlim=(-(lengths[0]+lengths[1]+lengths[2])*1.2, (lengths[0]+lengths[1]+lengths[2])*1.2), ylim=(-(lengths[0]+lengths[1]+lengths[2])*1.2, (lengths[0]+lengths[1]+lengths[2])*1.2))

    color_tails = {0: 'xkcd:lime', 1: 'xkcd:peach', 2: 'xkcd:sky blue'}
    marker_face = {0: 'xkcd:bright green', 1: 'xkcd:salmon', 2: 'xkcd:azure'}
    ax_pend_lines = [[], [], [], [], []]
 
    axes_v[position['energy_tot']].set_title("Energia Tot")
    axes_v[position['energy_k_p']].set_title("Energia Potenziale e Cinetica")
    axes_v[position['energy_tot']].set_xlim(0, tempo_simulazione)
    axes_v[position['energy_k_p']].set_xlim(0, tempo_simulazione)

    for p in range(n):
# lines from a mass to another
        ax_pend_lines[0].append(axes_v[position["motion"]].plot([], [], color='k', linestyle='-', linewidth=2, animated = True)[0])    
    # different for so it is better from a visual point of view
    for p in range(n):
# tail and points
        ax_pend_lines[1].append(axes_v[position["motion"]].plot([], [], 'o-',color = color_tails[p],markersize = 12, markerfacecolor = marker_face[p],linewidth=2, markevery=10000, markeredgecolor = 'k', animated = True)[0])
        ax_pend_lines[2].append(axes_v[position['polar']].plot([], [], 'o-',color = color_tails[p],markersize = 12, markerfacecolor = marker_face[p],linewidth=2, markevery=10000, markeredgecolor = 'k', animated = True)[0])

#    a = axes_v[position['energy_tot']].plot([],[] , 'o-',color = color_tails[2],markersize = 1, markerfacecolor = marker_face[2], linewidth=2, markeredgecolor = 'k', animated = True)[0] 
    ax_pend_lines[3].append(axes_v[position["energy_k_p"]].plot([], [], 'o-',color = color_tails[0],markersize = 1, markerfacecolor = marker_face[0],linewidth=2, markevery=10000, markeredgecolor = 'k', animated = True)[0])    
    ax_pend_lines[3].append(axes_v[position["energy_k_p"]].plot([], [], 'o-',color = color_tails[0],markersize = 1, markerfacecolor = marker_face[0],linewidth=2, markevery=10000, markeredgecolor = 'k', animated = True)[0])    
    ax_pend_lines[4].append(axes_v[position["energy_tot"]].plot([], [], 'o-',color = color_tails[0],markersize = 1, markerfacecolor = marker_face[0],linewidth=2, markevery=10000, markeredgecolor = 'k', animated = True)[0])    

    time_text = axes_v[position['motion']].text(0.02, 0.95, '', transform=axes_v[position['motion']].transAxes)
    time_text = axes_v[position['energy_tot']].text(0.02, 0.95, '', transform=axes_v[position['energy_tot']].transAxes)
    energy_text = axes_v[position['motion']].text(0.02, 0.90, '', transform=axes_v[position['motion']].transAxes)
    energy_text = axes_v[position['energy_k_p']].text(0.02, 0.90, '', transform=axes_v[position['energy_k_p']].transAxes)

    # first plot then set otherwise it doesn't work
    axes_v[position['polar']].set_theta_zero_location("S")
    axes_v[position['polar']].set_rmax(sum(lengths))

    fig.set_figheight(8)
    fig.set_figwidth(8)
     
    def init():
        for p in range(n):
            ax_pend_lines[0][p].set_data([], [])
            ax_pend_lines[1][p].set_data([], [])
            ax_pend_lines[2][p].set_data([], [])

#        ax_pend_lines[3][0].set_data([], [])
#        ax_pend_lines[3][1].set_data([], [])
#        ax_pend_lines[4].set_data([], [])

        time_text.set_text('')
        energy_text.set_text('')
        return ax_pend_lines[0]+ ax_pend_lines[1]+ ax_pend_lines[2] + ax_pend_lines[3] + ax_pend_lines[4] +[time_text, energy_text] 
         

    def animate(i):
        global t
        nonlocal u
        tails = [10, 10, 10]
        fps_jump = max(int((1/h)/framepersec), 1) # ogni quanto devo saltare di scrivere i frame per ottenere al piu' framepersec  foto in un secondo
        for x in range(fps_jump):
            u = f(f_n[n], u)
            t+=h
        u0 = u[0]

        x, y = get_xy_coords(u0)
        theta, r = get_polar_coords(u0)

        for k in range(n):
            x_plot[k].append(x[k])
            y_plot[k].append(y[k])
            r_plot[k].append(r[k])
            theta_plot[k].append(theta[k])

        t_plot.append(t)
        en_tot.append(energia_totale(u0, n))
        en_k.append(energia_cinetica(u0, n))
        en_p.append(energia_potenziale(u0, n))


        # line from the origin to the first mass
        ax_pend_lines[0][0].set_data([0, x[0]], [0, y[0]])
        #[i+1:max(1,i+1-tails[i]):-1]  
        for j in range(1, n):
        # line from the i-1 mass to the i mass
            ax_pend_lines[0][j].set_data([x[j-1], x[j]], [y[j-1], y[j]])

        ax_pend_lines[3][0].set_data(t_plot, en_k) 
        ax_pend_lines[3][1].set_data(t_plot, en_p) 
        ax_pend_lines[4][0].set_data(t_plot, en_tot) 
        axes_v[2].clear()
        axes_v[3].clear()
        axes_v[2].plot(t_plot, en_k)
        axes_v[2].plot(t_plot, en_p)
        axes_v[3].plot(t_plot, en_tot)
        ax_pend_lines[4][0].set_data(t_plot, en_tot) 
        for j in range(n):
            ax_pend_lines[1][j].set_data(x_plot[j][i+1:max(1, i+1-tails[j]):-1], y_plot[j][i+1:max(1,i+1-tails[j]):-1])
            ax_pend_lines[2][j].set_data(theta_plot[j][::-1], r_plot[j][::-1])
    
        time_text.set_text('time = %.1f' % (t))
        energy_text.set_text('energy = %.9f J' % en_tot[i])
        
        print(f"  {int(100*i/min(framepersec * tempo_simulazione, int(tempo_simulazione/h)))} % Processing", end="\r") 
        return (ax_pend_lines[0]+ ax_pend_lines[1]+ ax_pend_lines[2] + ax_pend_lines[3] + ax_pend_lines[4])

    
    anim = animation.FuncAnimation(fig, func = animate, init_func = init, interval=max(1000/framepersec, h*1000), frames = int(min(framepersec * tempo_simulazione, tempo_simulazione/h)), repeat = False, blit = True)
    anim.save(output)
    print(" Done             ")
#    plt.show()
    return anim


#dati iniziali di default
# condizioni iniziali
# angolo iniziale in gradi
#grad0_1, grad0_2, grad0_3 =  (135, 135, 135)
#thetas0  =  (grad0_1*2*np.pi / 360 ,grad0_2*2*np.pi / 360 ,grad0_3*2*np.pi / 360)
lengths  = np.array([1., 1., 1.])
masses   = np.array([1., 1., 1.])
grads0 = np.array([135, 135, 135])
thetas0 = grads0 * 2 * np.pi /360
omegas0_grad = np.zeros(3)
omegas0  = omegas0_grad * 2 * np.pi / 360
g = 9.81
h = 0.001
tempo_simulazione = 10
framepersec = 30

# dictionary to simplify life for input n other things
n_pend_string = {1: "single", 2: "double", 3: "triple"}
#f_anim_pendulum = {animate_single_pendulum, 2:animate_double_pendulum, 3:animate_triple_pendulum}
d_f_int = {1: runge_kutta4, 2: velocity_verlet, 3: trapezoide_implicito, 4:eulero_implicito, 5:eulero_semi_implicito, 6:eulero_esplicito, 7:stormer_verlet}
n_i =  input("Method of Numerical integration ? \n  [1] Runge Kutta 4 \n  2 Velocity Verlet \n  3 Implicit Verlet \n  4 Implicit Eulero \n  5 Semi-Implicit Eulero  \n  6 Explicit Eulero \n  7 Stormer Verlet \n")

if (n_i == ""): f_int = runge_kutta4
else: f_int = d_f_int[int(n_i)]
   
n_p = 3

y_n = input(f"Running with Default configuration? [Y/n] \n   N pendulum = {n_p} \n   time step = {h}s \n   theta_0 = {grads0}grad \n   l = {lengths}m \n   m = {masses}Kg \n   fps = {framepersec}s**-1 \n   time simulation = {tempo_simulazione}s \n   g = {g}m/s**2 \n").lower()
if (y_n == "n"):
    n_p = (input("n pendula to simulate? [1, 2, [3]] \n"))
    if( n_p in ["1", "2", "3"]):
        n_p = int(n_p) 
    elif (not n_p): n_p = 3
    else: exit("n pendola not supported atm.")

    for i in range(n_p):
        print(f"Pend n. {i+1} :")
        l = input(f" Length [{lengths[i]}] m: \n  ")
        if (l.lstrip('-').replace('.','',1).replace('e-','',1).replace('e','',1).isdigit()): lengths[i] = float(l)
        elif (l): 
            print("Input non valido.")
            exit()
        m = input(f" Mass [{masses[i]}] Kg: \n  ")
        if (m.lstrip('-').replace('.','',1).replace('e-','',1).replace('e','',1).isdigit()): masses[i] = float(m)
        elif (m): 
            print("Input non valido.")
            exit()
        theta = input(f" Initial theta [{grads0[i]}] Grad:  \n  ")
        if (theta.lstrip('-').replace('.','',1).replace('e-','',1).replace('e','',1).isdigit()): thetas0[i] =  float(theta)*2*np.pi / 360 
        elif (theta): 
            print("Input non valido.")
            exit()
        omega = input(f" Initial omega [{omegas0_grad[i]}] Grad/s:  \n  ")
        if (omega.lstrip('-').replace('.','',1).replace('e-','',1).replace('e','',1).isdigit()): omegas0[i] = float(omega)*2*np.pi / 360
        elif (omega): 
            print("Input non valido.")
            exit()
        print()


    gravity = input(f"Gravity [{g}] m/s**2 :\n")
    if (gravity.lstrip('-').replace('.','',1).replace('e-','',1).replace('e','',1).isdigit()): g = float(gravity)
    elif (gravity): 
        print("Input non valido.")
        exit()

    step = input(f"Default time steps [{h}] s :\n")
    if (step.lstrip('-').replace('.','',1).replace('e-','',1).replace('e','',1).isdigit()): h = float(step)
    elif (step): 
        print("Input non valido.")
        exit()

    time = input(f"Time simulation [{tempo_simulazione}] s :\n")
    if (time.lstrip('-').replace('.','',1).replace('e-','',1).replace('e','',1).isdigit()): tempo_simulazione = float(time)
    elif (time): 
        print("Input non valido.")
        exit()

    fps = input(f"Fps [{framepersec}] :  \n")
    if (fps.isdigit()): framepersec = int(fps)
    elif (fps): 
        print("Input non valido.")
        exit()
 
elif y_n != "y" and y_n:
        print("wrong input.")
        exit()

#f_int = [runge_kutta4, velocity_verlet, trapezoide_implicito, eulero_implicito, eulero_semi_implicito, eulero_esplicito, stormer_verlet] 
dict_mode = { 1: "angles",2:"velocities", 3: "masses", 4: "lengths", 0: "nothing"}
perturb = 1e-4
n_pend = 40
t_start = perf_counter()
mode = input("Select Mode: \n  [1] Single \n   2 The Butterfly Effect \n")
if mode == "1" or not mode: 
    animate_pendulum(f_int, f"{animate_pendulum.__name__}{f_int.__name__}_{n_pend_string[n_p]}.mp4", n_p)
    f_pendulum = animate_pendulum
elif mode == "2": 

    n_mode =  input("What to perturb?\n  [1] Angles 4 \n   2 Angular Velocities \n   3 Masses \n   4 Lengths \n   0 Nothing")
    if (n_mode.lstrip('-').isdigit()): n_mode = int(n_mode)
    elif (not n_mode): n_mode = 1
    else: 
        print("Input non valido.")
        exit()

    s_perturb = input("Module of perturbation? [grad | grad/s | Kg | m]\n ")
    if (s_perturb.lstrip('-').replace('.','',1).replace('e-','',1).replace('e','',1).isdigit()): perturb = float(s_perturb)
    elif (s_perturb): 
        print("Input non valido.")
        exit()
    butterfly_effect(f_int, f"{butterfly_effect.__name__}-perturb_{dict_mode[n_mode]}-{s_perturb}-{f_int.__name__}_{n_pend_string[n_p]}.mp4", n_p, n_pend, perturb, n_mode)
print(f"Running {n_pend_string[n_p]} pendulum with {f_int.__name__} propagation...")
t_end = perf_counter()
#t_prec = 2
print(f"Tempo di esecuzione: {t_end - t_start: .4}")
