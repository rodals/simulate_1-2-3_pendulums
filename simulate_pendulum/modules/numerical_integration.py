########## Methods for numerical integration ##########
import numpy as np
import copy
import pendulum_functions as pf

def forward_euler(pend):
    f = pend.f_accel()
    q = pend.get_q()
    dot_q = pend.get_omegas()
    t = pend.time
    h = pend.h_step
    masses = pend.masses
    lengths = pend.lengths
    g = pend.g

    inc_q, inc_dot_q = f(q, dot_q, t, lengths, masses, g)
    q += inc_q*h
    dot_q += inc_dot_q*h
    return q, dot_q

def backward_euler(pend):
    f = pend.f_accel()
    q = pend.get_q()
    dot_q = pend.get_omegas()
    t = pend.time
    h = pend.h_step
    masses = pend.masses
    lengths = pend.lengths
    g = pend.g
    s = 1e-12
    c_q, c_dot_q = np.zeros(pend.type_pend), np.zeros(pend.type_pend)
    diff_q, diff_dot_q = np.ones(pend.type_pend)*2*s, np.ones(pend.type_pend)*2*s
    while ((np.abs(diff_q)>s).any() and (np.abs(diff_dot_q)>s).any()):
        diff_q, diff_dot_q = c_q, c_dot_q
        c_q, c_dot_q =  f(q + c_q, dot_q + c_dot_q, t+h, lengths, masses, g)
        c_q *= h
        c_dot_q *= h
        diff_q -= c_q
        diff_dot_q -= c_dot_q
    q += c_q
    dot_q += c_dot_q
    return q, dot_q
 
def semi_implicit_euler(pend):
    f = pend.f_accel()
    q = pend.get_q()
    dot_q = pend.get_omegas()
    t = pend.time
    h = pend.h_step
    masses = pend.masses
    lengths = pend.lengths
    g = pend.g

    dot_q += f(q, dot_q, t, lengths, masses, g)[1]*h
    q += f(q, dot_q, t, lengths, masses, g)[0]*h
    return q, dot_q

def symplectic_euler(pend):
    f = pend.f_accel()
    q = pend.get_q()
    p = pend.get_p()
    t = pend.time
    h = pend.h_step
    masses = pend.masses
    lengths = pend.lengths
    g = pend.g
    f_q = {pf.f_single: pf.single_d_q_H, pf.f_double: pf.double_d_q_H, pf.f_triple: pf.triple_d_q_H}
    f_p = {pf.f_single: pf.single_d_p_H, pf.f_double: pf.double_d_p_H, pf.f_triple: pf.triple_d_p_H}
    s = 1e-12

# if non separable implicit, otherwise it automatically exit:  p_{n+1} = p_n - h * d_{q_i} H( p_{n+1}, q_n ) 
    
    p  -= implicit(f_q[f], q, p, t, lengths, masses, g, h, s, "p")
#  q_{n+1} = q_n - h * d_{p_i} H( p_{n+1}, q_n )
    q  += f_p[f]( q, p, t, lengths, masses, g)*h
    return q, p

def stormer_verlet(pend):
    f = pend.f_accel()
    q = pend.get_q()
    p = pend.get_p()
    t = pend.time
    h = pend.h_step
    masses = pend.masses
    lengths = pend.lengths
    g = pend.g
    s = 1e-12
    f_q = {pf.f_single: pf.single_d_q_H, pf.f_double: pf.double_d_q_H, pf.f_triple: pf.triple_d_q_H}
    f_p = {pf.f_single: pf.single_d_p_H, pf.f_double: pf.double_d_p_H, pf.f_triple: pf.triple_d_p_H}

    p  -= implicit(f_q[f], q, p, t, lengths, masses, g, h/2, s, "p")
    q  += f_p[f]( q, p, t, lengths, masses, g)*h/2
    q  += implicit(f_p[f], q, p, t, lengths, masses, g, h/2, s, "q")
    p -= f_q[f](q, p, t, lengths, masses, g)*h/2
    return q, p

def velocity_verlet(pend):
    f = pend.f_accel()
    q = pend.get_q()
    dot_q = pend.get_omegas()
    t = pend.time
    h = pend.h_step
    masses = pend.masses
    lengths = pend.lengths
    g = pend.g

    q_1 = copy.deepcopy(q)
    dot_q_1 = copy.deepcopy(dot_q)
    q_1 = q + dot_q*h  + f(q, dot_q, t, lengths, masses, g)[1]*h*h/2
    dot_q_1 = dot_q + (f(q, dot_q, t, lengths, masses, g)[1] + f(q_1, dot_q_1, t+h, lengths, masses, g)[1])*h/2
    return q_1, dot_q_1

def two_step_adams_bashforth(pend, pend_b):
    f = pend.f_accel()
    q_b = pend_b.get_q()
    dot_q_b = pend_b.get_omegas()
    q = pend.get_q()
    dot_q = pend.get_omegas()
    t = pend.time
    h = pend.h_step
    masses = pend.masses
    lengths = pend.lengths
    g = pend.g

    # multistep, need a second point
    if t == 0:
        q, dot_q = runge_kutta4(pend)
        pend.increment_time()
    q_1  = q + (3./2)*h*f(q, dot_q, t, lengths, masses, g)[0] - (1./2)*h*f(q_b, dot_q_b, pend_b.time, lengths, masses, g)[0]
    dot_q_1 = dot_q + (3./2)*h*f(q, dot_q, t, lengths, masses, g)[1] - (1./2)*h*f(q_b, dot_q_b, pend_b.time, lengths, masses, g)[1]
    return ((q_1, dot_q_1), (q, dot_q))

def crank_nicolson(pend):
    f = pend.f_accel()
    q = pend.get_q()
    dot_q = pend.get_omegas()
    t = pend.time
    h = pend.h_step
    masses = pend.masses
    lengths = pend.lengths
    g = pend.g
    s = 1e-12
    c_q, c_dot_q = np.zeros(pend.type_pend), np.zeros(pend.type_pend)
    diff_q, diff_dot_q = np.ones(pend.type_pend)*2*s, np.ones(pend.type_pend)*2*s
    q_0, dot_q_0 = f(q, dot_q, t, lengths, masses, g)
    q   += q_0 * h/2
    dot_q   += dot_q_0 * h/2
    while ((np.abs(diff_q)>s).any() and (np.abs(diff_dot_q)>s).any()):
        diff_q, diff_dot_q = c_q, c_dot_q
        c_q, c_dot_q =  f(q + c_q, dot_q + c_dot_q, t+h, lengths, masses, g)
        c_q *= h/2
        c_dot_q *= h/2
        diff_q -= c_q
        diff_dot_q -= c_dot_q
    q += c_q
    dot_q += c_dot_q
    return q, dot_q
           
def runge_kutta4(pend):
    f = pend.f_accel()
    q = pend.get_q()
    dot_q = pend.get_omegas()
    t = pend.time
    h = pend.h_step
    masses = pend.masses
    lengths = pend.lengths
    g = pend.g

    k1_q, k1_dot_q = f(q, dot_q, t, lengths, masses, g)
    k2_q, k2_dot_q = f(q + k1_q*h/2, dot_q + k1_dot_q*h/2, t + h/2, lengths, masses, g)
    k3_q, k3_dot_q = f(q + k2_q*h/2, dot_q + k2_dot_q*h/2, t + h/2, lengths, masses, g)
    k4_q, k4_dot_q = f(q + k3_q*h, dot_q + k3_dot_q*h, t + h, lengths, masses, g)
    q = (q + (k1_q + 2*k2_q + 2*k3_q + k4_q)*h/6.0)
    dot_q = (dot_q + (k1_dot_q + 2*k2_dot_q + 2*k3_dot_q + k4_dot_q)*h/6.0)
    return q, dot_q

#### usefull function 
# function to calculate functions implicitly
def implicit(f, q, p, t, lengths, masses, g, h_inc,  s, p_or_q):
    diff = np.ones(len(lengths))*2*s
    d = np.zeros(len(lengths))
    count = 0
    count_max = 500
    while( (np.abs(diff) > s).any() and count < count_max):
        if(p_or_q =="q"): d1 = f(q+d, p, t, lengths, masses, g)*h_inc
        elif(p_or_q =="p"):  d1 = f(q, p-d, t, lengths, masses, g)*h_inc
        diff = d1 -  d
        d = copy.deepcopy(d1)
        count+=1
    return d

