########## Methods for numerical integration ##########

def forward_euler(f, u):
    u[0] = u[0] + f(u[0] , t)*h
    return u

def backward_euler(f, u):
    s = 1e-12
    c = np.zeros(6)
    diff = np.array([2*s, 2*s, 2*s, 2*s, 2*s, 2*s])
    while (np.abs(diff)>s).any():
        diff = c
        c =  f(u[0] + c, t+h)*h
        diff -= c
    u[0] += c        
    return u

def semi_implicit_euler(f, u):
    u[0][3:6] += f(u[0] , t)[3:6]*h
    u[0][0:3] += f(u[0] , t)[0:3]*h
    return u

def symplectic_euler(f, u):
    f_q = {f_single: single_d_q_H, f_double: double_d_q_H, f_triple: triple_d_q_H}
    f_p = {f_single: single_d_p_H, f_double: double_d_p_H, f_triple: triple_d_p_H}
    q = u[0][0:3]
    p = u[0][3:6]
    s = 1e-12
# if non separable implicit, otherwise it automatically exit:  p_{n+1} = p_n - h * d_{q_i} H( p_{n+1}, q_n ) 
    p  -= implicit(f_q[f], q, p, h, s, "p")
#  q_{n+1} = q_n - h * d_{p_i} H( p_{n+1}, q_n )
    q  += f_p[f]( q, p)*h
    u[0][0:3] = q
    u[0][3:6] = p
    return u

def stormer_verlet(f, u):
    f_q = {f_single: single_d_q_H, f_double: double_d_q_H, f_triple: triple_d_q_H}
    f_p = {f_single: single_d_p_H, f_double: double_d_p_H, f_triple: triple_d_p_H}
    q = u[0][0:3]
    p = u[0][3:6]
    s = 1e-12
    p  -= implicit(f_q[f], q, p, h/2, s, "p")

    dq1  = f_p[f]( q, p)*h/2
    q  += implicit(f_p[f], q, p, h/2, s, "q")
    q += dq1

    p -= f_q[f](q, p)*h/2
    u[0][0:3] = q
    u[0][3:6] = p
    return u

def velocity_verlet(f, u):
    y_1 = copy.deepcopy(u[0])
    y_1[0:3] = u[0][0:3] + u[0][3:6]*h  + f(u[0], t)[3:6]*h*h/2
    y_1[3:6] = u[0][3:6] + (f(u[0], t)[3:6] + f(y_1, t+h)[3:6])*h/2
    u[0] = copy.deepcopy(y_1)
    return u

def two_step_adams_bashforth(f, u):
    global t
    # multistep, need a second point
    if t == 0:
        u[1] = runge_kutta4(f,u)[0]
        t+=h
    temp = u[1] + (3./2)*h*f(u[1], t) - (1./2)*h*f(u[0], t-h)
    u[0] = u[1]
    u[1] = copy.deepcopy(temp)
    return u

def crank_nicolson(f, u):
    s = 1e-12
    c = np.zeros(6)
    diff = np.array([2*s, 2*s, 2*s, 2*s, 2*s, 2*s])
    u[0] += f(u[0], t)*h/2
    while (np.abs(diff)>s).any():
        diff = c
        c =  f(u[0] + c, t+h)*h/2
        diff -= c
    u[0] += c        
    return u
           
def runge_kutta4(pend):
    f = pend.f_accel()
    u = pend.get_u()
    t = pend.time
    h = pend.h_step
    masses = pend.masses
    lengths = pend.lengths
    g = pend.g

    k1 = f(u, t, lengths, masses, g)*h
    k2 = f(u + k1/2, t +h/2, lengths, masses, g)*h
    k3 = f(u + k2/2 , t + h/2, lengths, masses, g)*h
    k4 = f(u + k3 , t + h, lengths, masses, g)*h
    u = (u + (k1 + 2*k2 + 2*k3 + k4)/6.0)
    return u[0:3], u[3:6]

