# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 18:31:59 2022

@author: Rafael Rivero de Nicolás
"""

from numpy import array, linspace, hstack#, reshape, zeros,

# from numpy.linalg import norm

from ODE_PDE_Solvers import Cauchy_Problem_V2

import LB_Temporal_Schemes as ts

import matplotlib.pyplot as plt

from LB_Physics_Problems import N_Bodies_Function

from matplotlib import rc # LaTeX tipography
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.rc('text', usetex=True); plt.rc('font', family='serif')

import matplotlib
matplotlib.rc('xtick', labelsize=18)
matplotlib.rc('ytick', labelsize=18)

# %% Cálculo

''' The code between ◘◘◘◘ should be editable by the user '''
''' ◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘ '''
SELECTED_SCHEME = 4

tf = 50

Delta_t = [0.1]   # Δt for different simulations

U_0_1 = array([2, 2, 0, 0.5, 0, 0])
U_0_2 = array([-2, -2, 0, -0.5, 0, 0])
U_0_3 = array([0, 0, 0, 0, 0, 0])



# U_0_1 = array([1.1547/2, 0, 0, 1.25*0.5, 1.25*0.866, 0]) # tf = 60
# U_0_2 = array([-1.1547/2, 0, 0, 1.25*0.5, -1.25*0.886, -0])
# U_0_3 = array([0, 1, 0, -1.25*1, 0, 0])
''' ◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘ '''

Temporal_schemes_available = {1:ts.Euler,
                              2:ts.Inverse__Euler,
                              3:ts.Crank__Nicolson,
                              4:ts.RK4,
                              5:ts.Leap_Frog}

Nb = 3; Nc=3
# U_0 = linspace(0,Nb*Nc*2-1,Nb*Nc*2)

# %% Pre-Computing

U_0 = hstack((U_0_1, U_0_2, U_0_3))

scheme = Temporal_schemes_available[SELECTED_SCHEME]

U = {}; time_domain = {}

# %% Simulations

for dt in Delta_t:

    N = int( tf/dt )

    time_domain[scheme.__name__+'__dt=' + str(dt)] = linspace( 0, tf, N+1 )

    print('Temporal partition used Δt = ', str(dt))

    U[scheme.__name__+'__dt=' + str(dt)] = Cauchy_Problem_V2( F = N_Bodies_Function, U_0 = U_0, time_domain = time_domain[scheme.__name__+'__dt=' + str(dt)], Temporal_scheme = scheme )

    print('\n\n\n')

# U = Cauchy_Problem_V2(N_Bodies_Function, U_0, time_domain, Temporal_scheme=ts.Leap_frog)

# %% Plots

colours = ['blue', 'red', 'black', 'orange', 'magenta', 'green', 'grey', 'cyan', 'yellow']

i = 0

for key in U:

    fig, ax1 = plt.subplots(1,1, figsize=(9,6), constrained_layout='true')
    # ax1.set_xlim(0,tf);
    ax1.set_title('Numeric Scheme: '+scheme.__name__, fontsize=20)
    ax1.grid();
    ax1.set_xlabel(r'$x$',fontsize=20); ax1.set_ylabel(r'$y$',fontsize=20);

    x = U[key][0,:]
    y = U[key][1,:]

    ax1.scatter( x, y, c=colours[i], s=7, label=r'Primer cuerpo')
    ax1.scatter( x[-1], y[-1], c=colours[i], s=30)
    # ax2.plot( x, z, c=colours[i], label=r'$\Delta t$ = '+key[len(scheme.__name__)+2:])

    i = i+1

    x = U[key][6,:]
    y = U[key][7,:]

    ax1.scatter( x, y, c=colours[i], s=7, label=r'Segundo cuerpo')
    ax1.scatter( x[-1], y[-1], c=colours[i], s=30)
    # ax2.plot( x, z, c=colours[i], label=r'$\Delta t$ = '+key[len(scheme.__name__)+2:])

    i = i+1

    x = U[key][12,:]
    y = U[key][13,:]

    ax1.scatter( x, y, c=colours[i], s=7, label=r'Tercer cuerpo')
    ax1.scatter( x[-1], y[-1], c=colours[i], s=30)
    # ax2.plot( x, z, c=colours[i], label=r'$\Delta t$ = '+key[len(scheme.__name__)+2:])

    i = i+1

    ax1.legend(loc=1, fancybox=True, edgecolor="black", ncol = 3, fontsize=16)
    plt.show()
    # fig.savefig('H5'+key+'.pdf', transparent = True, bbox_inches="tight")

    # New figure

    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.grid()

    x = U[key][0,:]
    y = U[key][1,:]

    ax.scatter(x, y, time_domain[key], c = 'r', s = 5)
    ax.scatter( x[-1], y[-1], time_domain[key][-1],  c='r', s=30)

    x = U[key][6,:]
    y = U[key][7,:]

    ax.scatter(x, y, time_domain[key], c = 'blue', s = 5)
    ax.scatter( x[-1], y[-1], time_domain[key][-1],  c='blue', s=30)

    x = U[key][12,:]
    y = U[key][13,:]

    ax.scatter(x, y, time_domain[key], c = 'black', s = 5)
    ax.scatter( x[-1], y[-1], time_domain[key][-1],  c='black', s=30)

    ax.set_title('3D Scatter Plot', fontsize=20)

    # Set axes label
    ax.set_xlabel(r'$x$', labelpad=20, fontsize=20)
    ax.set_ylabel(r'$y$', labelpad=20, fontsize=20)
    ax.set_zlabel(r'$t$', labelpad=20, fontsize=20)
    plt.show()

    # fig.savefig('H5_3D_'+key+'.pdf', transparent = True, bbox_inches="tight")

# ax2.legend(loc=2, fancybox=True, edgecolor="black", ncol = 3, fontsize=16)
# fig.savefig('H5_'+scheme.__name__+'_'+str(tf)+'.pdf', transparent = True, bbox_inches="tight")


# %% Demostración de que funciona esta vaina

# Nb = 5; Nc = 3

# U_0 = linspace(0,Nb*Nc*2-1,Nb*Nc*2)


# Us = reshape(U_0, (Nb, Nc*2))
# r = transpose(Us[:,0:Nc])
# v = transpose(Us[:,Nc:])

# # r[:,:] = r[:,:] + 100 # Debería cambiar el valor de Us

# # r = r[:,:] + 100 # No cambia el valor de Us

# # r = r + 100 # No cambia el valor de Us
