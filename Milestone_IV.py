# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 16:09:37 2022

@author: Rafael Rivero de Nicolás
"""


from numpy import hstack, linspace
# from math import pi

# from LB_Temporal_Schemes import Absolute_Stability_Region
from LB_Physics_Problems import Undamped_Armonic_Oscilator
from ODE_PDE_Solvers import Cauchy_Problem_V2

import LB_Temporal_Schemes as ts


# from mpmath import findroot

import matplotlib.pyplot as plt

from matplotlib import rc # LaTeX tipography
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.rc('text', usetex=True); plt.rc('font', family='serif')

import matplotlib
matplotlib.rc('xtick', labelsize=18)
matplotlib.rc('ytick', labelsize=18)

# %% Definition

r_0 = 0.5; v_0 = 0.5 # Initial position and velocity, respectively.

Differential_Operator = Undamped_Armonic_Oscilator # [Undamped Armonic Oscilator [1D]]

# Delta_t = [1.5, 0.5, 0.1] # Different Δt used for the simulation
Delta_t = [0.1, 0.08, 0.05] # For Euler and Inverse Euler
# Delta_t = [0.5, 0.25, 0.1] # For Leap Frog scheme
# Delta_t = [0.1] # For Leap Frog soluciones espúreas


tf = 30 # Final time of the simulation

Temporal_schemes_available = {1:ts.Euler,
                              2:ts.Inverse__Euler,
                              3:ts.Crank__Nicolson,
                              4:ts.RK4,
                              5:ts.Leap_Frog}

Temporal_scheme = Temporal_schemes_available[2]

# First parameters reorganization
Initial_conditions = hstack((r_0,v_0)); print('Initial State Vector: U_0 = ', Initial_conditions, '\n\n\n')
U = {}; time_domain = {}


# %% Numeric Simulation

for dt in Delta_t:

    time_domain[dt] = linspace(0, tf, int(tf/dt)+1 )

    U[dt] = Cauchy_Problem_V2(Differential_Operator, Initial_conditions, time_domain[dt], Temporal_scheme)


# %% Stability Region call

# ts.Absolute_Stability_Region(Temporal_scheme, lst = Delta_t) # User's Numeric Jacobian should be used for this purpose due to the use of complex values. It can be found in LB_Math_Functions

# %% Plots

colours = ['blue', 'red', 'black', 'orange', 'magenta', 'green', 'grey', 'cyan', 'yellow']

i = 0

fig, axes = plt.subplots(2,1, figsize=(9,6), constrained_layout='true')
ax1 = axes[0]; ax2 = axes[1]
ax1.set_xlim(0,tf); ax2.set_xlim(0,tf)
ax1.set_title('Numeric Scheme: '+Temporal_scheme.__name__, fontsize=20)
ax1.grid(); ax2.grid()
ax2.set_xlabel(r'$t$',fontsize=20)
ax1.set_ylabel(r'$x$',fontsize=20); ax2.set_ylabel(r'$\dot{x}$',fontsize=20)

for key in U:

    x = time_domain[key]
    y = U[key][0,:]
    z = U[key][1,:]

    ax1.plot( x, y, c=colours[i], label=r'$\Delta t$ = '+str(key))
    ax2.plot( x, z, c=colours[i], label=r'$\Delta t$ = '+str(key))

    i = i+1

ax1.legend(loc=2, fancybox=True, edgecolor="black", ncol = 3, fontsize=16)
ax2.legend(loc=2, fancybox=True, edgecolor="black", ncol = 3, fontsize=16)
plt.show()
# fig.savefig('H4_'+Temporal_scheme.__name__+'_'+str(tf)+'.pdf', transparent = True, bbox_inches="tight")
