import numpy as np  
import matplotlib.pyplot as plot  
from scipy.integrate import odeint 

def AP_model(k=8, a=0.15, eps_0=0.002, mu_1=0.2, mu_2=0.3):
    def eps_fun(u, v):
        return eps_0 + mu_1*v/(u+mu_2)

    def fun(y, t):
        u, v = y
        eps = eps_fun(u,v)
        dydt = [-k*u*(u-a)*(u-1)-u*v, eps*(-v-k*u*(u-a-1))]
        return dydt
    return fun

def u_to_mV(u):
    return 100 * u - 80

def t_to_ms(t):
    return 12.9 * t

def pictures(t, solution):
    fig = plot.figure(1, figsize=(10, 12))

    ax1 = fig.add_subplot(221)
    ax1.plot(t, solution[:, 0],'k',linewidth=2)
    ax1.set_title('Adimensional action potential u')
    ax1.set_xlabel('t [-]')
    ax1.set_ylabel('u [-]')

    ax2 = fig.add_subplot(222)
    ax2.plot(t, solution[:, 1],'b',linewidth=2)
    ax2.set_title('Adimensional gating variale v')
    ax2.set_xlabel('t [-]')
    ax2.set_ylabel('v [-]')

    ax3 = fig.add_subplot(223)
    ax3.plot(solution[:, 0], solution[:, 1],'r',linewidth=2)
    ax3.set_title('Phase diagram')
    ax3.set_xlabel('u [-]')
    ax3.set_ylabel('v [-]')

    E = u_to_mV(solution[:, 0])
    T = t_to_ms(t)

    ax4 = fig.add_subplot(224)
    ax4.set_title('Dimensional action potential E')
    ax4.plot(T, E,'g',linewidth=2)
    ax4.set_xlabel('T [ms]')
    ax4.set_ylabel('E [mV]')

    plot.show()


def run():
    t_0 = 0
    t_end = 40
    discretization = (t_end - t_0) * 1000
    t = np.linspace(t_0, t_end, discretization)

    y0 = (0.17, 0)

    solution = odeint(AP_model(), y0, t)

    pictures(t, solution)

run()    