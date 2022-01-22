import matplotlib.pyplot as plt
import numpy as np
import copy
from  time import perf_counter
from matplotlib.animation import FuncAnimation

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
    

def animate_triple_pendulum(f, output):
    global t
    n_pend = 3
    t = 0
    x1_plot = []
    x2_plot = []
    x3_plot = []
    y1_plot = []
    y2_plot = []
    y3_plot = []
    r1_plot = []
    r2_plot = []
    r3_plot = []
    theta1_plot = []
    theta2_plot = []
    theta3_plot = []

    u0 = np.array([thetas0[0], thetas0[1], thetas0[2], omegas0[0], omegas0[1], omegas0[2]])
    um1 = np.array([thetas0[0], thetas0[1], thetas0[2], omegas0[0], omegas0[1], omegas0[2]])
    u = np.array([u0, um1])

    fig = plt.figure()
    ax_pend = plt.subplot(221)
    ax_polar = fig.add_subplot(222, projection='polar')
    ax_polar.set_theta_offset(np.pi/2)
    ax_en_tot = fig.add_subplot(223)
    ax_en_k_p = fig.add_subplot(224)
    #ax_lissajous = fig.add_subplot(225)

    ax_pend.set_aspect('equal', adjustable='box')
    ax_pend.axis('off')
    ax_pend.set(xlim=(-(lengths[0]+lengths[1]+lengths[2])*1.2, (lengths[0]+lengths[1]+lengths[2])*1.2), ylim=(-(lengths[0]+lengths[1]+lengths[2])*1.2, (lengths[0]+lengths[1]+lengths[2])*1.2))

# lines from a mass to another
    line1, = ax_pend.plot([], [], color='k', linestyle='-', linewidth=2)    
    line2, = ax_pend.plot([], [], color='k', linestyle='-', linewidth=2)    
    line3, = ax_pend.plot([], [], color='k', linestyle='-', linewidth=2)    

# tail and points
    line1_tail, = ax_pend.plot([], [], 'o-',color = '#d2eeff',markersize = 12, markerfacecolor = '#0077BE',linewidth=2, markevery=10000, markeredgecolor = 'k')   # line for Earth
    line2_tail, = ax_pend.plot([], [], 'o-',color = '#ff3bd2',markersize = 12, markerfacecolor = '#f66338',linewidth=2, markevery=10000, markeredgecolor = 'k')   # line for Earth
    line3_tail, =  ax_pend.plot([], [], 'o-',color = '#af3bd2',markersize = 12, markerfacecolor = '#af6338',linewidth=2, markevery=10000, markeredgecolor = 'k')   # line for Earth
    time_text = ax_pend.text(0.02, 0.95, '', transform=ax_pend.transAxes)
    energy_text = ax_pend.text(0.02, 0.90, '', transform=ax_pend.transAxes)

# graphs 
    en_tot = [energia_totale(u0, n_pend)]
    en_k = [energia_cinetica(u0, n_pend)]
    en_p = [energia_potenziale(u0, n_pend)]
    scatti_xy = [get_xy_coords(u0)]
    theta, r = get_polar_coords(u0)
    

    fig.set_figheight(8)
    fig.set_figwidth(8)
     
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        line1_tail.set_data([], [])
        line2_tail.set_data([], [])
        line3_tail.set_data([], [])
        time_text.set_text('')
        energy_text.set_text('')
        return line, time_text, energy_text

    def animate(i):
        global t
        nonlocal u
        tail1 = 10
        tail2 = 10
        tail3 = 10
        fps_jump = max(int((1/h)/framepersec), 1) # ogni quanto devo saltare di scrivere i frame per ottenere al piu' framepersec  foto in un secondo
        for x in range(fps_jump):
            u = f(f_triple, u)
            t+=h
        u0 = u[0]

        x, y = get_xy_coords(u0)

        x1_plot.append(x[0])
        x2_plot.append(x[1])
        x3_plot.append(x[2])
        y1_plot.append(y[0])
        y2_plot.append(y[1])
        y3_plot.append(y[2])

        theta, r = get_polar_coords(u0)
        theta1_plot.append(theta[0])
        theta2_plot.append(theta[1])
        theta3_plot.append(theta[2])



        r1_plot.append(r[0])
        r2_plot.append(r[1])
        r3_plot.append(r[2])

        scatti_xy.append([x, y])
        en_tot.append(energia_totale(u0, n_pend))
        en_k.append(energia_cinetica(u0, n_pend))
        en_p.append(energia_potenziale(u0, n_pend))

        # line from the origin to the first mass
        line1.set_data([0, x[0]], [0, y[0]])
        # line from the first mass to the second mass
        line2.set_data([x[0], x[1]], [y[0], y[1]])
        # line from the second to the third mass
        line3.set_data([x[1], x[2]], [y[1], y[2]])
        line1_tail.set_data(x1_plot[i+1:max(1,i+1-tail1):-1], y1_plot[i+1:max(1,i+1-tail1):-1])
        line2_tail.set_data(x2_plot[i+1:max(1,i+1-tail2):-1], y2_plot[i+1:max(1,i+1-tail2):-1])
        line3_tail.set_data(x3_plot[i+1:max(1,i+1-tail3):-1], y3_plot[i+1:max(1,i+1-tail3):-1])
        time_text.set_text('time = %.1f' % (t))
        energy_text.set_text('energy = %.9f J' % en_tot[i+1])
        
        ax_en_tot.clear()
        ax_en_k_p.clear()
        ax_polar.clear()
#        ax_lissajous.clear()
        ax_polar.set_theta_zero_location("S")

        ax_polar.plot(theta1_plot, r1_plot)
        ax_polar.plot(theta2_plot, r2_plot)
        ax_polar.plot(theta3_plot, r3_plot)

#        ax_lissajous.plot(theta1_plot, theta2_plot)

        ax_en_tot.plot(en_tot, label='Energia Totale')
        ax_en_k_p.plot(en_k, label='Energia Cinetica')
        ax_en_k_p.plot(en_p, label='Energia Potenziale')
        print(f"  {int(100*i/min(framepersec * tempo_simulazione, int(tempo_simulazione/h)))} % Processing", end="\r") 

    
    anim = FuncAnimation(fig, func = animate, interval=max(1000/framepersec, h*1000), frames = int(min(framepersec * tempo_simulazione, tempo_simulazione/h)), repeat = False, blit =False)
    anim.save(output)
    print(" Done             ")
#    plt.show()
    return anim

def animate_double_pendulum(f, output):
    global t
    n_pend = 2
    t = 0
    x1_plot = []
    x2_plot = []
    y1_plot = []
    y2_plot = []
    r1_plot = []
    r2_plot = []
    theta1_plot = []
    theta2_plot = []

    u0 = np.array([thetas0[0], thetas0[1], 0, omegas0[0], omegas0[1], 0])
    um1 = np.array([thetas0[0], thetas0[1], 0, omegas0[0], omegas0[1], 0])
    u = np.array([u0, um1])

    fig = plt.figure()
    ax_pend = plt.subplot(221)
#    ax_lissajous = fig.add_subplot(222)
    ax_en_tot = fig.add_subplot(223)
    ax_en_k_p = fig.add_subplot(224)
    ax_polar = fig.add_subplot(222, projection='polar')
    ax_polar.set_theta_offset(np.pi/2)

    ax_pend.set_aspect('equal', adjustable='box')
    ax_pend.axis('off')
    ax_pend.set(xlim=(-(lengths[0]+lengths[1])*1.2, (lengths[0]+lengths[1])*1.2), ylim=(-(lengths[0]+lengths[1])*1.2, (lengths[0]+lengths[1])*1.2))

# lines from a mass to another
    line1, = ax_pend.plot([], [], color='k', linestyle='-', linewidth=2)    
    line2, = ax_pend.plot([], [], color='k', linestyle='-', linewidth=2)    

# tail and points
    line1_tail, = ax_pend.plot([], [], 'o-',color = '#d2eeff',markersize = 12, markerfacecolor = '#0077BE',linewidth=2, markevery=10000, markeredgecolor = 'k')   # line for Earth
    line2_tail, = ax_pend.plot([], [], 'o-',color = '#ff3bd2',markersize = 12, markerfacecolor = '#f66338',linewidth=2, markevery=10000, markeredgecolor = 'k')   # line for Earth
    time_text = ax_pend.text(0.02, 0.95, '', transform=ax_pend.transAxes)
    energy_text = ax_pend.text(0.02, 0.90, '', transform=ax_pend.transAxes)

# graphs 
    en_tot = [energia_totale(u0, n_pend)]
    en_k = [energia_cinetica(u0, n_pend)]
    en_p = [energia_potenziale(u0, n_pend)]
    scatti_xy = [get_xy_coords(u0)]
    theta, r = get_polar_coords(u0)
    

    fig.set_figheight(8)
    fig.set_figwidth(8)
     
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line1_tail.set_data([], [])
        line2_tail.set_data([], [])
        time_text.set_text('')
        energy_text.set_text('')
        return line1, time_text, energy_text

    def animate(i):
        global t
        nonlocal u
        tail1 = 10
        tail2 = 10
        fps_jump = max(int((1/h)/framepersec), 1) # ogni quanto devo saltare di scrivere i frame per ottenere al piu' framepersec  foto in un secondo
        for x in range(fps_jump):
            u = f(f_double, u)
            t+=h

        u0 = u[0]
        x, y = get_xy_coords(u0)

        x1_plot.append(x[0])
        x2_plot.append(x[1])
        y1_plot.append(y[0])
        y2_plot.append(y[1])

        theta, r = get_polar_coords(u0)
        theta1_plot.append(theta[0])
        theta2_plot.append(theta[1])

        r1_plot.append(r[0])
        r2_plot.append(r[1])

        scatti_xy.append([x, y])
        en_tot.append(energia_totale(u0, n_pend))
        en_k.append(energia_cinetica(u0, n_pend))
        en_p.append(energia_potenziale(u0, n_pend))

        # line from the origin to the first mass
        line1.set_data([0, x[0]], [0, y[0]])
        # line from the first mass to the second mass
        line2.set_data([x[0], x[1]], [y[0], y[1]])
        # line from the second to the third mass
        line1_tail.set_data(x1_plot[i+1:max(1,i+1-tail1):-1], y1_plot[i+1:max(1,i+1-tail1):-1])
        line2_tail.set_data(x2_plot[i+1:max(1,i+1-tail2):-1], y2_plot[i+1:max(1,i+1-tail2):-1])
        time_text.set_text('time = %.1f' % (t))
        energy_text.set_text('energy = %.9f J' % en_tot[i+1])
        
        ax_en_tot.clear()
        ax_en_k_p.clear()
        ax_polar.clear()
#        ax_lissajous.clear()
        ax_polar.set_theta_zero_location("S")

        ax_polar.plot(theta1_plot, r1_plot)
        ax_polar.plot(theta2_plot, r2_plot)

#        ax_lissajous.plot(theta1_plot, theta2_plot)

        ax_en_tot.plot(en_tot, label='Energia Totale')
        ax_en_k_p.plot(en_k, label='Energia Cinetica')
        ax_en_k_p.plot(en_p, label='Energia Potenziale')
        print(f"  {int(100*i/min(framepersec * tempo_simulazione, int(tempo_simulazione/h)))} % Processing", end="\r") 

    
    anim = FuncAnimation(fig, func = animate, interval=max(1000/framepersec, h*1000), frames = int(min(framepersec * tempo_simulazione, tempo_simulazione/h)), repeat = False, blit =False)
    anim.save(output)
    print(" Done             ")
#    plt.show()
    return anim


def animate_single_pendulum(f, output):
    global t
    n_pend = 1
    t = 0
    x1_plot = []
    y1_plot = []
    r1_plot = []
    theta1_plot = []

    u0 = np.array([thetas0[0], 0, 0, omegas0[0], 0, 0])
    um1 = np.array([thetas0[0], 0, 0, omegas0[0], 0, 0])
    u = np.array([u0, um1])

    fig = plt.figure()
    ax_pend = plt.subplot(221)
    ax_polar = fig.add_subplot(222, projection='polar')
    ax_polar.set_theta_offset(np.pi/2)
    ax_en_tot = fig.add_subplot(223)
    ax_en_k_p = fig.add_subplot(224)

    ax_pend.set_aspect('equal', adjustable='box')
    ax_pend.axis('off')
    ax_pend.set(xlim=(-(lengths[0])*1.2, (lengths[0])*1.2), ylim=(-(lengths[0])*1.2, (lengths[0])*1.2))

# lines from a mass to another
    line1, = ax_pend.plot([], [], color='k', linestyle='-', linewidth=2)    

# tail and points
    line1_tail, = ax_pend.plot([], [], 'o-',color = '#d2eeff',markersize = 12, markerfacecolor = '#0077BE',linewidth=2, markevery=10000, markeredgecolor = 'k')   # line for Earth
    time_text = ax_pend.text(0.02, 0.95, '', transform=ax_pend.transAxes)
    energy_text = ax_pend.text(0.02, 0.90, '', transform=ax_pend.transAxes)

# graphs 
    en_tot = [energia_totale(u0, n_pend)]
    en_k = [energia_cinetica(u0, n_pend)]
    en_p = [energia_potenziale(u0, n_pend)]
    scatti_xy = [get_xy_coords(u0)]
    theta, r = get_polar_coords(u0)
    

    fig.set_figheight(8)
    fig.set_figwidth(8)
     
    def init():
        line1.set_data([], [])
        line1_tail.set_data([], [])
        time_text.set_text('')
        energy_text.set_text('')
        return line1, time_text, energy_text

    def animate(i):
        global t
        nonlocal u
        tail1 = 10
        fps_jump = max(int((1/h)/framepersec), 1) # ogni quanto devo saltare di scrivere i frame per ottenere al piu' framepersec  foto in un secondo
        for x in range(fps_jump):
            u = f(f_single, u)
            t+=h
        u0 = u[0]

        x, y = get_xy_coords(u0)

        x1_plot.append(x[0])
        y1_plot.append(y[0])

        theta, r = get_polar_coords(u0)
        theta1_plot.append(theta[0])

        r1_plot.append(r[0])

        scatti_xy.append([x, y])
        en_tot.append(energia_totale(u0, n_pend))
        en_k.append(energia_cinetica(u0, n_pend))
        en_p.append(energia_potenziale(u0, n_pend))

        # line from the origin to the first mass
        line1.set_data([0, x[0]], [0, y[0]])

        line1_tail.set_data(x1_plot[i+1:max(1,i+1-tail1):-1], y1_plot[i+1:max(1,i+1-tail1):-1])
        time_text.set_text('time = %.1f' % (t))
        energy_text.set_text('energy = %.9f J' % en_tot[i+1])
        
        ax_en_tot.clear()
        ax_en_k_p.clear()
        ax_polar.clear()
        ax_polar.set_theta_zero_location("S")

        ax_polar.plot(theta1_plot, r1_plot)
        ax_en_tot.plot(en_tot, label='Energia Totale')
        ax_en_k_p.plot(en_k, label='Energia Cinetica')
        ax_en_k_p.plot(en_p, label='Energia Potenziale')

    
    anim = FuncAnimation(fig, func = animate, interval=max(1000/framepersec, h*1000), frames = int(min(framepersec * tempo_simulazione, tempo_simulazione/h)), repeat = False, blit =False)
    anim.save(output)
    print(" Done             ")
#    plt.show()
    return anim

#dati iniziali di default



# condizioni iniziali
# angolo iniziale in gradi
#grad0_1, grad0_2, grad0_3 =  (135, 135, 135)
#thetas0  =  (grad0_1*2*np.pi / 360 ,grad0_2*2*np.pi / 360 ,grad0_3*2*np.pi / 360)
lengths  = [1., 1., 1.]
masses   = [1., 1., 1.]
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
f_anim_pendulum = {1:animate_single_pendulum, 2:animate_double_pendulum, 3:animate_triple_pendulum}
d_f_int = {1: runge_kutta4, 2: velocity_verlet, 3: trapezoide_implicito, 4:eulero_implicito, 5:eulero_semi_implicito, 6:eulero_esplicito, 7:stormer_verlet}
n_i =  input("Method of Numerical integration ? \n  [1] Runge Kutta 4 \n  2 Velocity Verlet \n  3 Implicit Verlet \n  4 Implicit Eulero \n  5 Semi-Implicit Eulero  \n  6 Explicit Eulero \n  7 Stormer Verlet \n")

if (n_i == ""): f_int = runge_kutta4
else: f_int = d_f_int[int(n_i)]
   
n_p = 3

y_n = input(f"Running with Default configuration? [Y/n] \n   N pendulum = {n_p} \n   time step = {h}s \n   theta_0 = {grads0}grad \n   l = {lengths}m \n   m = {masses}Kg \n   fps = {framepersec}s**-1 \n   time simulation = {tempo_simulazione}s \n   g = {g}m/s**2 \n").lower()
if (y_n == "n"):
    n_p = (input("n pendula to simulate? [1, 2, [3]]   "))
    if( n_p in ["1", "2", "3"]):
        n_p = int(n_p) 
    elif (not n_p): n_p = 3
    else: exit("n pendola not supported atm.")

    for i in range(n_p):
        print(f"Pend n. {i+1} :")
        l = input(f" Lenght [{lengths[i]}] m:   ")
        if (l.lstrip('-').replace('.','',1).replace('e-','',1).replace('e','',1).isdigit()): lengths[i] = float(l)
        elif (l): 
            print("Input non valido.")
            exit()
        m = input(f" Mass [{masses[i]}] Kg:   ")
        if (m.lstrip('-').replace('.','',1).replace('e-','',1).replace('e','',1).isdigit()): masses[i] = float(m)
        elif (m): 
            print("Input non valido.")
            exit()
        theta = input(f" Initial theta [{grads0[i]}] Grad:   ")
        if (theta.lstrip('-').replace('.','',1).replace('e-','',1).replace('e','',1).isdigit()): thetas0[i] =  float(theta)*2*np.pi / 360 
        elif (theta): 
            print("Input non valido.")
            exit()
        omega = input(f" Initial omega [{omegas0_grad[i]}] Grad/s:   ")
        if (omega.lstrip('-').replace('.','',1).replace('e-','',1).replace('e','',1).isdigit()): omegas0[i] = float(omega)*2*np.pi / 360
        elif (omega): 
            print("Input non valido.")
            exit()
        print()


    gravity = input(f"Gravity [{g}] m/s**2 :   ")
    if (gravity.lstrip('-').replace('.','',1).replace('e-','',1).replace('e','',1).isdigit()): g = float(gravity)
    elif (gravity): 
        print("Input non valido.")
        exit()

    step = input(f"Default time steps [{h}] s :   ")
    if (step.lstrip('-').replace('.','',1).replace('e-','',1).replace('e','',1).isdigit()): h = float(step)
    elif (step): 
        print("Input non valido.")
        exit()

    time = input(f"Time simulation [{tempo_simulazione}] s :   ")
    if (time.lstrip('-').replace('.','',1).replace('e-','',1).replace('e','',1).isdigit()): tempo_simulazione = float(time)
    elif (time): 
        print("Input non valido.")
        exit()

    fps = input(f"Fps [{framepersec}] :   ")
    if (fps.isdigit()): framepersec = int(fps)
    elif (fps): 
        print("Input non valido.")
        exit()
 
elif y_n != "y" and y_n:
        print("wrong input.")
        exit()

#f_int = [runge_kutta4, velocity_verlet, trapezoide_implicito, eulero_implicito, eulero_semi_implicito, eulero_esplicito, stormer_verlet] 
f_pendulum = f_anim_pendulum[n_p]
t_start = perf_counter()
print(f"Running {f_int.__name__}...")
f_pendulum(f_int, f"{f_int.__name__}_{n_pend_string[n_p]}.mp4")
t_end = perf_counter()
#t_prec = 2
print(f"Tempo di esecuzione: {t_end - t_start: .4}")
