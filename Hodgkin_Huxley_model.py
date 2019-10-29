import numpy as np
import math

g_L = 0.003
g_K = 0.36
g_Na = 1.2
E_L = -54.387
E_K = -77
E_Na = 50
C_m = 0.01

initial_values = [-60, 0.317, 0.0529, 0.596]

def alpha_m(V):
    return 0.1*(V + 40)/(1 - np.exp(-0.1*(V + 40)))

def beta_m(V):
    return 4*np.exp(-0.0556*(V + 65))

def alpha_n(V):
    return 0.01*(V + 55)/(1 - np.exp(-0.1*(V + 55)))

def beta_n(V):
    return 0.125*np.exp(-0.0125*(V + 65))

def alpha_h(V):
    return 0.07*np.exp(-0.05*(V + 65))

def beta_h(V):
    return 1/(1 + np.exp(-0.1*(V + 35)))

def HH_model_derivs(y, t, params):
    V, n, m, h = y
    g_L, g_K, g_Na, E_L, E_K, E_Na, C_m, I_e_func = params
    derivs = [
        (I_e_func(t) - g_K*math.pow(n, 4)*(V - E_K) - g_Na*math.pow(m, 3)*h*(V - E_Na) - g_L*(V - E_L))/C_m,
        alpha_n(V)*(1 - n) - beta_n(V)*n,
        alpha_m(V)*(1 - m) - beta_m(V)*m,
        alpha_h(V)*(1 - h) - beta_h(V)*h
    ]
    return derivs

            