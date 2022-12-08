# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 22:01:27 2022

@author: Rafael Rivero de Nicolás
"""

from LB_Math_Functions import Numeric_Jacobian #, Newton_Raphson
from ODE_PDE_Solvers import Embedded_RK_Application, Cauchy_Problem_V2
from LB_Physics_Problems import Autonomous_F, R3BP_Earth_Moon


from numpy import zeros, array, linspace #, size
from numpy.linalg import eigvals
from numpy.random import rand
# from scipy.optimize import newton

from matplotlib import rc # LaTeX tipography
import matplotlib.pyplot as plt
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
plt.rc('text', usetex=True); plt.rc('font', family='serif')

import matplotlib
matplotlib.rc('xtick', labelsize=18)
matplotlib.rc('ytick', labelsize=18)


# %% Initialitation

U0 = zeros([6, 5]) # 6 coordinates per body and 5 Lagrange points for 2 main Bodies

# U0[:,0] = [ 0.8, 0.6, 0., 0., 0., 0.  ]   # Lagrange points calculation
# U0[:,1] = [ 0.8, -0.6, 0., 0., 0., 0.  ]
# U0[:,2] = [ -0.1, 0.0, 0., 0., 0., 0.  ]
# U0[:,3] = [ 0.1, 0.0, 0., 0., 0., 0.  ]
# U0[:,4] = [ 1.1, 0.0, 0., 0., 0., 0.  ]
# # Sol = newton(Autonomous_F, U0[:,4]) # No converge bien xD
# L1 = mth.Newton_Raphson(Autonomous_F, x_i=U0[:,3])
# L2 = mth.Newton_Raphson(Autonomous_F, x_i=U0[:,4])
# L3 = mth.Newton_Raphson(Autonomous_F, x_i=U0[:,2])
# L4 = mth.Newton_Raphson(Autonomous_F, x_i=U0[:,0])
# L5 = mth.Newton_Raphson(Autonomous_F, x_i=U0[:,1])

L = {} # Dictionary that will contain Lagrange points
J = {}; lambdas = {}

L[1] = array([0.8369151258197125, 0. , 0. , 0. , 0. , 0. ])
L[2] = array([1.1556821654078693, 0. , 0. , 0. , 0. , 0. ])
L[3] = array([-1.0050626458062681, 0. , 0. , 0. , 0. , 0. ])
L[4] = array([0.48784941440000085, 0.8660254037844383 , 0. , 0. , 0. , 0. ])
L[5] = array([0.48784941440000085, -0.8660254037844383 , 0. , 0. , 0. , 0. ])

for (key, value) in L.items():

    U0[:,key-1] = value

    J[key] = Numeric_Jacobian(Autonomous_F, U0[:,key-1])

    lambdas[key] = eigvals(J[key])

for key in L:
    print("For Lagrange point L"+str(key)+"::    ")
    for i, value in enumerate(lambdas[key]):
        print("  Re(λ_"+str(i)+") = "+ str(value.real) )


# %% Simulation and plotting

tf = 300; nt = 12000; dt = tf/nt

time_domain = linspace(0, tf, nt)

Uper = U0+rand(U0.shape[0], U0.shape[1])*1E-4; Uper[2,:] = 0


for j, v in enumerate(U0.transpose()): # i will be the index of the columns

    U = Cauchy_Problem_V2(R3BP_Earth_Moon, Uper[:,j], time_domain) # Default RK4

    U2, dt_min_reached = Embedded_RK_Application(Uper[:,j], R3BP_Earth_Moon, time_domain, name="RK87", tolerance = 1E-10, dt_min = 0.001)

    colours = ['blue', 'red', 'black', 'orange', 'magenta', 'green', 'grey', 'cyan', 'yellow']

    # i = 0

    # fig, ax = plt.subplots(1,1, figsize=(10,7), constrained_layout='true')
    # # ax1.set_xlim(0,tf); ax2.set_xlim(0,tf)
    # ax.set_title('Orbit around Lagrange point L'+str(j+1), fontsize=20)
    # ax.grid(); ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    # ax.set_xlabel(r'$x$',fontsize=20);
    # ax.set_ylabel(r'$y$',fontsize=20);

    # x = U[0,:]
    # y = U[1,:]

    # ax.scatter( U0[0,j], U0[1,j], c="magenta", s=30 )
    # ax.scatter( x, y, c=colours[i], s=7, label=r'$\Delta t_{o}$ = '+str(dt) )


    i = 0
    fig, ax = plt.subplots(1,1, figsize=(10,7), constrained_layout='true')
    # ax1.set_xlim(0,tf); ax2.set_xlim(0,tf)
    ax.set_title('Orbit around Lagrange point L'+str(j+1), fontsize=20)
    ax.grid();
    ax.set_xlabel(r'$x$',fontsize=20);
    ax.set_ylabel(r'$y$',fontsize=20);

    x = U[0,:]
    y = U[1,:]

    ax.scatter( 0, 0, c="red", s=50 )
    ax.scatter( 1, 0, c="blue", s=50 )
    ax.scatter( U0[0,j], U0[1,j], c="magenta", s=50 )
    ax.scatter( x, y, c='black', s=7, label=r'$\Delta t_{o}$ = '+str(dt) )


    i = 0
    fig, ax = plt.subplots(1,1, figsize=(10,7), constrained_layout='true')
    # ax1.set_xlim(0,tf); ax2.set_xlim(0,tf)
    ax.set_title('Orbit with ERK 87 around Lagrange point L'+str(j+1), fontsize=20)
    ax.grid();
    ax.set_xlabel(r'$x$',fontsize=20);
    ax.set_ylabel(r'$y$',fontsize=20);

    xx = U2[0,:]
    yy = U2[1,:]

    ax.scatter( 0, 0, c="red", s=50 )
    ax.scatter( 1, 0, c="blue", s=50 )
    ax.scatter( U0[0,j], U0[1,j], c="magenta", s=50 )
    ax.scatter( xx, yy, c='black', s=7, label=r'$\Delta t_{o}$ = '+str(dt) )
    plt.show()


    if j == 3 or j == 4:

        fig, ax = plt.subplots(1,1, figsize=(10,7), constrained_layout='true')
        ax.set_title('Orbit around Lagrange point L'+str(j+1), fontsize=20)
        ax.grid();
        ax.set_xlabel(r'$x$',fontsize=20);
        ax.set_ylabel(r'$y$',fontsize=20);

        x = U[0,:]
        y = U[1,:]

        ax.scatter( U0[0,j], U0[1,j], c="magenta", s=50 )
        ax.scatter( x, y, c='black', s=7, label=r'$\Delta t_{o}$ = '+str(dt) )


        i = 0
        fig, ax = plt.subplots(1,1, figsize=(10,7), constrained_layout='true')
        # ax1.set_xlim(0,tf); ax2.set_xlim(0,tf)
        ax.set_title('Orbit with ERK 87 around Lagrange point L'+str(j+1), fontsize=20)
        ax.grid();
        ax.set_xlabel(r'$x$',fontsize=20);
        ax.set_ylabel(r'$y$',fontsize=20);

        xx = U2[0,:]
        yy = U2[1,:]

        ax.scatter( U0[0,j], U0[1,j], c="magenta", s=50 )
        ax.scatter( xx, yy, c='black', s=7, label=r'$\Delta t_{o}$ = '+str(dt) )
        plt.show()


