# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 23:46:33 2022

@author: Rafael Rivero de Nicolás
"""

from numpy import linspace, absolute, array, meshgrid, zeros, matmul #, exp, sqrt, size,

import LB_Math_Functions as mth



# from scipy.optimize import fsolve

from scipy.optimize import newton
import matplotlib.pyplot as plt

# %% Numeric Temporal Schemes

def Euler(X, t, dt, Function):
    '''
    Parameters
    ----------
    X : Array
        State vector of the system in instant t: U_(n).
    t : Array
        Time instant in which Function is being evaluated.
    dt : Float
        Time step used during the simulation, also called Δt.
    Function : Function previosuly defined
        This function satisfies the problem dX/dt = Function(X,t) and must be an input argument.

    Returns
    -------
    Array
        U_n1 is the state vector of the system in instant t+dt.
        It is also the vector that satisfies: U_(n+1) = U_(n) + dt * Function(U_(n),t).

    '''

    return X + dt*Function(X,t)


def Inverse__Euler(X, t, dt, Function):
    '''
    Parameters
    ----------
    X : Array
        State vector of the system in instant t: X = U_(n).
    t : Array
        Time instant in which Function is being evaluated.
    dt : Float
        Time step used during the simulation, also called Δt.
    Function : Function previosuly defined
        This function satisfies the problem dX/dt = Function(X,t) and must be an input argument.

    Returns
    -------
    Array
        U_n1 is the state vector of the system in instant t+dt.
        It is also the vector that satisfies: U_(n+1) = U_(n) + dt * Function(U_(n+1),t+dt).

    '''

    def Inverse__Euler__Operator(U_n1):

        return U_n1 - X - dt * Function(U_n1, t)

    # U_n1 = mth.Newton_Raphson(Inverse__Euler__Operator, x_i=X) # Rafa's Function

    # U_n1 = fsolve(Inverse__Euler__Operator, X)

    # U_n1 = newton(Inverse__Euler__Operator, X)

    return newton(Inverse__Euler__Operator, X)


def RK4(X, t, dt, Function):
    '''
    Parameters
    ----------
    X : Array
        State vector of the system in instant t: U_(n).
    t : Array
        Time instant in which Function is being evaluated.
    dt : Float
        Time step used during the simulation, also called Δt.
    Function : Function previosuly defined
        This function satisfies the problem dX/dt = Function(X,t) and must be an input argument.

    Returns
    -------
    Array
        U_n1 is the state vector of the system in instant t+dt.
        It is also the vector that satisfies: U_(n+1) = U_(n) + dt * ( k1 + 2*k2 + 2*k3 + k4 ) / 6.

    '''

    k1 = Function( X, t )

    k2 = Function( X + dt * k1/2, t + dt/2 )

    k3 = Function( X + dt * k2/2, t + dt/2 )

    k4 = Function( X + dt *k3,    t + dt   )

    U_n1 = X + dt * ( k1 + 2*k2 + 2*k3 + k4 ) / 6

    return U_n1


def Crank__Nicolson(X, t, dt, Function):
    '''
    Parameters
    ----------
    X : Array
        State vector of the system in instant t.
    t : Array
        Time instant in which Function is being evaluated.
    dt : Float
        Time step used during the simulation, also called Δt.
    Function : Function previosuly defined
        This function satisfies the problem dX/dt = Function(X,t) and must be an input argument.

    Returns
    -------
    Array
        U_n1 is the state vector of the system in instant t+dt.
        It is also the vector that satisfies: U_(n+1) = X + dt/2 * ( Function(X,t) + Function(U_(n+1), t+dt)).

    '''

    def  Crank_Nicolson_Operator(U_n1):
        return  U_n1 - X - dt/2 * ( Function(X,t) + Function(U_n1,t+dt) )

    # U_n1 = fsolve(Crank_Nicolson_Operator, X) # Scipy Function

    # U_n1 = mth.Newton_Raphson(Crank_Nicolson_Operator, x_i=X)

    return  newton(Crank_Nicolson_Operator, X)


def Leap_Frog(X, X_1, t, dt, Function):
    '''
    Parameters [NOT UPDATED]
    ----------
    X : Array
        State vector of the system in instant t.
    X_1 : Array
        State vector of the system in instant t-dt.
    t : Array
        Time instant in which Function is being evaluated.
    dt : Float
        Time step used during the simulation, also called Δt.
    Function : Function previosuly defined
        This function satisfies the problem dX/dt = Function(X,t) and must be an input argument.

    Returns
    -------
    Array
        U_n1 is the state vector of the system in instant t+dt.
        It is also the vector that satisfies: U_(n+1) = X_1 + 2*dt * ( Function(X,t) ).

    '''

    return X_1 + 2*dt*Function(X,t) # U_n1

# %% Stability Regions

def stab_F(U, t):

    x = linspace(-5,5,200)
    y = linspace(-5j,5j,200)

    x, y = meshgrid(x,-y)

    w_mesh = x + y

    return U*w_mesh

######
# def Stab_Inv_Euler(X, t, dt, Function, w):

#     def sInverse__Euler__Operator(U_n1):

#         return U_n1 - X - dt * Function(U_n1, t, w)

#     Un = mth.Newton_Raphson(sInverse__Euler__Operator, x_i = X)

#     return Un

# def Stab_Crank__Nicolson(X, t, dt, Function, w):

#     def  sCrank_Nicolson_Operator(U_n1):
#         return  U_n1 - X - dt/2 * ( Function(X,t,w) + Function(U_n1,t+dt,w) )

#     return  mth.Newton_Raphson(sCrank_Nicolson_Operator, x_i=X)


# def Stab_Euler(X, t, dt, Function, w):
#     return X + dt*Function(X,t,w)

# def Stab_RK4(X, t, dt, Function, w):

#     k1 = Function( X, t, w )

#     k2 = Function( X + dt * k1/2, t + dt/2, w )

#     k3 = Function( X + dt * k2/2, t + dt/2, w )

#     k4 = Function( X + dt *k3,    t + dt,   w   )

#     U_n1 = X + dt * ( k1 + 2*k2 + 2*k3 + k4 ) / 6

#     return U_n1
#######

def Stability_Polynomial(scheme):
    if scheme.__name__=="Euler":
        r = Euler(1, 0, 1, stab_F)
    elif scheme.__name__=="RK4":
        r = RK4(1, 0, 1, stab_F)
    elif scheme.__name__=="Inverse__Euler":
        r = Inverse__Euler(1, 0, 1, stab_F)
    elif scheme.__name__=="Crank__Nicolson":
        r = Crank__Nicolson(1, 0, 1, stab_F)

    return r

def Absolute_Stability_Region(Temporal_Scheme, lst):


    Reg_Stability = absolute(array(Stability_Polynomial(Temporal_Scheme)))

    x = linspace(-5,5,200)
    y = linspace(-5,5,200)

    colours = ['blue', 'red', 'black', 'orange', 'magenta', 'green', 'grey', 'cyan', 'yellow']

    Contour_plot_figure, ax = plt.subplots(nrows=1,ncols=1,figsize=[9,(9/1.618)], constrained_layout=True)
    ax.contourf(x, y, Reg_Stability, levels = [0, 1],  colors=['#C0C0C0']) #levels = [0.5, 1, 1.5]


    if Temporal_Scheme.__name__ == "Euler":
        ax.set_xlim(-2.5, 0.5)
        ax.set_ylim(-1.5, 1.5)
    elif Temporal_Scheme.__name__ == "RK4":
        ax.set_xlim(-4, 1)
        ax.set_ylim(-3, 3)
    elif Temporal_Scheme.__name__ == "Inverse__Euler":
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-1.5, 1.5)
    else:
        ax.set_xlim(x[0],x[-1])
        ax.set_ylim(y[0],y[-1])

    it=0

    for element in lst:

        ax.scatter(0, element,  marker="o", c=colours[it], label=r'$\Delta t$ = '+str(lst[it]))

        it = it + 1

    ax.set_title('Absolute Stability Region of '+Temporal_Scheme.__name__, fontsize=16)
    ax.set_xlabel(r"$\Re(\omega)$", fontsize=18)
    ax.set_ylabel(r"$\Im(\omega)$", fontsize=18)
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.text(0.035,0.9,'Stability Region', fontsize=18, transform=ax.transAxes, bbox=dict(facecolor='#C0C0C0', edgecolor='black'))
    ax.grid()
    plt.show()

    return Reg_Stability

# %% Embedded Runge Kutta


def ERK_Selection(name="RK21"): # Embbeded Runge Kutta Butcher array selection


    if   (name=="HeunEuler21"):
        q = [2,1]
        Ne = 2

        a = zeros([Ne, Ne-1]); b = zeros(Ne); bs = zeros(Ne); c = zeros(Ne)

        c[:] = array([ 0., 1. ])
        #-------------------------------------------------------------------------------------
        a[0,:] = array([  0. ])
        a[1,:] = array([  1. ])
        #-------------------------------------------------------------------------------------
        b[:] = array([ 1./2, 1./2 ])
        bs[:] = array([ 1.,    0.  ])

        return a, b, bs, c, q


    elif   (name=="RK21"):
        q = [2,1]
        Ne = 3

        a = zeros([Ne, Ne-1]); b = zeros(Ne); bs = zeros(Ne); c = zeros(Ne)

        c[:] = array([ 0., 0.5, 1. ])
        #-------------------------------------------------------------------------------------
        a[0,:] = array([  0., 0. ])
        a[1,:] = array([  1./2, 0. ])
        a[2,:] = array([  1./256,  255./256	])
        #-------------------------------------------------------------------------------------
        b[:] = array([ 1./256,	255./256,	0. ])
        bs[:] = array([ 1./512,	255./256,	1./512 ])

        return a, b, bs, c, q


    elif   (name=="BogackiShampine"):
        q = [3,2]
        Ne = 4

        a = zeros([Ne, Ne-1]); b = zeros(Ne); bs = zeros(Ne); c = zeros(Ne)

        c[:] = array([ 0., 1./2, 3./4, 1. ])
        #-------------------------------------------------------------------------------------
        a[0,:] = array([  0., 0., 0.            ])
        a[1,:] = array([ 1./2, 0., 0.           ])
        a[2,:] = array([ 0.,	3./4, 0.    	])
        a[3,:] = array([ 2./9,	1./3,	4./9 	])
        #-------------------------------------------------------------------------------------
        b[:] = array([ 2./9,	1./3,	4./9,	0. ])
        bs[:] = array([ 7./24,	1./4,	1./3,	1./8 ])

        return a, b, bs, c, q



    elif     (name=="DOPRI54"):
        q = [5,4]
        Ne = 7

        a = zeros([Ne, Ne-1]); b = zeros(Ne); bs = zeros(Ne); c = zeros(Ne)

        c[:] = array([ 0., 1./5, 3./10, 4./5, 8./9, 1., 1. ])
        #-------------------------------------------------------------------------------------
        a[0,:] = array([          0.,           0.,           0.,         0.,           0.,     0. ])
        a[1,:] = array([      1./5  ,           0.,           0.,         0.,           0.,     0. ])
        a[2,:] = array([      3./40 ,        9./40,           0.,         0.,           0.,     0. ])
        a[3,:] = array([     44./45 ,      -56./15,        32./9,         0.,           0.,     0. ])
        a[4,:] = array([ 19372./6561, -25360./2187,  64448./6561,  -212./729,           0.,     0. ])
        a[5,:] = array([  9017./3168,    -355./33 ,  46732./5247,    49./176, -5103./18656,     0. ])
        a[6,:] = array([    35./384 ,           0.,    500./1113,   125./192, -2187./6784 , 11./84 ])
        #-------------------------------------------------------------------------------------
        b[:] = array([ 35./384   , 0.,   500./1113,  125./192,  -2187./6784  ,  11./84  ,     0.])
        bs[:] = array([5179./57600, 0., 7571./16695,  393./640, -92097./339200, 187./2100, 1./40 ])


        return a, b, bs, c, q


    elif (name=="CashKarp"):
        q = [5,4]
        Ne = 6

        a = zeros([Ne, Ne-1]); b = zeros(Ne); bs = zeros(Ne); c = zeros(Ne)

        c[:] = array([ 0., 1./5, 3./10, 3./5, 1., 7./8 ])
           #-------------------------------------------------------------------------------------
        a[0,:] = array([ 0.,          0.,       0.,         0.,            0. ])
        a[1,:] = array([ 1./5,        0.,       0.,         0.,            0. ])
        a[2,:] = array([ 3./40,       9./40,    0.,         0.,            0. ])
        a[3,:] = array([ 3./10,      -9./10,    6./5,       0.,            0. ])
        a[4,:] = array([ -11./54,     5./2,    -70./27,     35./27,        0. ])
        a[5,:] = array([ 1631./55296, 175./512, 575./13824, 44275./110592, 253./4096 ])
           #-------------------------------------------------------------------------------------

        b[:] = array([    37./378, 0.,     250./621,     125./594,         0., 512./1771])
        bs[:] = array([2825./27648, 0., 18575./48384, 13525./55296, 277./14336,     1./4 ])

        return a, b, bs, c, q



    elif (name=="Fehlberg54"):
        q = [5,4]
        Ne = 6

        a = zeros([Ne, Ne-1]); b = zeros(Ne); bs = zeros(Ne); c = zeros(Ne)

        c[:] = array([ 0., 1./4, 3./8, 12./13, 1., 1./2 ])
           #-------------------------------------------------------------------------------------
        a[0,:] = array([ 0.        ,   0.       ,  0.        ,    0.     ,            0. ])
        a[1,:] = array([ 1./4      ,   0.       ,  0.        ,    0.     ,            0. ])
        a[2,:] = array([ 3./32     ,   9./32    ,  0.        ,    0.     ,            0. ])
        a[3,:] = array([ 1932./2197, -7200./2197, 7296./2197 ,    0.     ,            0. ])
        a[4,:] = array([ 439./216  ,  -8.       , 3680./513  , -845./4104,            0. ])
        a[5,:] = array([ -8./27    ,   2.       , -3544./2565, 1859./4104,       -11./40 ])
           #-------------------------------------------------------------------------------------
        b[:] = array([ 16./135, 0., 6656./12825, 28561./56430, -9./50, 2./55])
        bs[:] = array([25./216, 0., 1408./2565, 2197./4104, -1./5 , 0. ])

        return a, b, bs, c, q



    elif (name=="Fehlberg87"):
        q = [8,7]
        Ne = 13

        a = zeros([Ne, Ne-1]); b = zeros(Ne); bs = zeros(Ne); c = zeros(Ne)

        c[:] = array([ 0., 2./27, 1./9, 1./6, 5./12, 1./2, 5./6, 1./6, 2./3 , 1./3,   1., 0., 1.])
           #-------------------------------------------------------------------------------------
        a[0,:] = array([ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        a[1,:] = array([ 2./27, 0., 0., 0., 0., 0., 0.,  0., 0., 0., 0., 0.])
        a[2,:] = array([ 1./36 , 1./12, 0., 0., 0., 0., 0.,  0.,0., 0., 0., 0.])
        a[3,:] = array([ 1./24 , 0., 1./8 , 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        a[4,:] = array([ 5./12, 0., -25./16, 25./16., 0., 0., 0., 0., 0., 0., 0., 0.])
        a[5,:] = array([ 1./20, 0., 0., 1./4, 1./5, 0., 0.,0., 0., 0., 0., 0.])
        a[6,:] = array([-25./108, 0., 0., 125./108, -65./27, 125./54, 0., 0., 0., 0., 0., 0.])
        a[7,:] = array([ 31./300, 0., 0., 0., 61./225, -2./9, 13./900, 0., 0., 0., 0., 0.])
        a[8,:] = array([ 2., 0., 0., -53./6, 704./45, -107./9, 67./90, 3., 0., 0., 0., 0.])
        a[9,:] = array([-91./108, 0., 0., 23./108, -976./135, 311./54, -19./60, 17./6, -1./12, 0., 0., 0.])
        a[10,:] = array([ 2383./4100, 0., 0., -341./164, 4496./1025, -301./82, 2133./4100, 45./82, 45./164, 18./41, 0., 0.])
        a[11,:] = array([ 3./205, 0., 0., 0., 0., -6./41, -3./205, -3./41, 3./41, 6./41, 0., 0.])
        a[12,:] = array([ -1777./4100, 0., 0., -341./164, 4496./1025, -289./82, 2193./4100, 51./82, 33./164, 19./41, 0.,  1.])
           #-------------------------------------------------------------------------------------
        b[:] = array([ 41./840, 0., 0., 0., 0., 34./105, 9./35, 9./35, 9./280, 9./280, 41./840, 0., 0.])
        bs[:] = array([ 0., 0., 0., 0., 0., 34./105, 9./35, 9./35, 9./280, 9./280, 0., 41./840, 41./840])

        return a, b, bs, c, q


    elif (name=="Verner65"):
        q = [6,5]
        Ne = 8

        a = zeros([Ne, Ne-1]); b = zeros(Ne); bs = zeros(Ne); c = zeros(Ne)

        c[:] = array([ 0., 1./6, 4./15, 2./3, 5./6, 1., 1./15, 1. ])
           #-------------------------------------------------------------------------------------
        a[0,:] = array([ 0. , 0.,  0., 0.,  0. ,  0. ,   0. ])
        a[1,:] = array([ 1./6 , 0.,  0., 0.,  0. ,  0. ,   0. ])
        a[2,:] = array([ 4./75 , 16./75,  0., 0.,  0. ,  0. ,   0. ])
        a[3,:] = array([ 5./6 , -8./3,  5./2, 0.,  0. ,  0. ,   0. ])
        a[4,:] = array([ -165./64 , 55./6, -425./64, 85./96,  0. ,  0. ,   0. ])
        a[5,:] = array([ 12./5, -8., 4015./612, -11./36,  88./255 ,  0. ,   0. ])
        a[6,:] = array([ -8263./15000, 124./75, -643./680, -81./250, 2484./10625 ,  0. ,   0. ])
        a[7,:] = array([ 3501./1720 , -300./43, 297275./52632, -319./2322, 24068./84065 ,  0. ,   3850./26703 ])
           #-------------------------------------------------------------------------------------
        b[:] = array([ 3./40 , 0.,  875./2244, 23./72, 264./1955,    0., 125./11592, 43./616 ])
        bs[:] = array([13./160, 0., 2375./5984,  5./16,    12./85, 3./44,         0.,      0. ])

        return a, b, bs, c, q



    elif (name=="RK65"):
        q = [6,5]
        Ne = 8

        a = zeros([Ne, Ne-1]); b = zeros(Ne); bs = zeros(Ne); c = zeros(Ne)

        c[:] = array([ 0., 1./10, 2./9, 3./7, 3./5, 4./5, 1., 1. ])
           #-------------------------------------------------------------------------------------
        a[0,:] = array([ 0.             ,  0.        ,  0.                , 0.               ,  0.              ,  0.             ,   0. ])
        a[1,:] = array([ 1./10          ,  0.        ,  0.                , 0.               ,  0.              ,  0.             ,   0. ])
        a[2,:] = array([ -2./81         ,  20./81    ,  0.                , 0.               ,  0.              ,  0.             ,   0. ])
        a[3,:] = array([ 615./1372      ,  -270./343 ,  1053./1372        , 0.               ,  0.              ,  0.             ,   0. ])
        a[4,:] = array([ 3243./5500     ,  -54./55   ,  50949./71500      , 4998./17875      ,  0.              ,  0.             ,   0. ])
        a[5,:] = array([ -26492./37125  ,  72./55    ,  2808./23375       , -24206./37125    ,  338./459        ,  0.             ,   0. ])
        a[6,:] = array([ 5561./2376     , -35./11    ,  -24117./31603     , 899983./200772   ,  -5225./1836     ,  3925./4056     ,   0. ])
        a[7,:] = array([ 465467./266112 , -2945./1232,  -5610201./14158144, 10513573./3212352,  -424325./205632 ,  376225./454272 ,   0. ])
           #-------------------------------------------------------------------------------------
        b[:] = array([ 61./864   , 0., 98415./321776, 16807./146016 , 1375./7344 , 1375./5408, -37./1120, 1./10])
        bs[:] = array([ 821./10800, 0., 19683./71825 , 175273./912600,  395./3672 ,  785./2704,     3./50,   0. ])

        return a, b, bs, c, q


    elif (name=="RK87"):
        q = [8,7]
        Ne = 13

        a = zeros([Ne, Ne-1]); b = zeros(Ne); bs = zeros(Ne); c = zeros(Ne)

        c[:] = array([ 0., 1./18, 1./12, 1./8, 5./16, 3./8, 59./400, 93./200, 5490023248./9719169821, 13./20, 1201146811./1299019798, 1., 1.])
        #-------------------------------------------------------------------------------------
        a[0,:] = array([ 0.                     , 0.   ,   0.   , 0.                       , 0., 0., 0., 0., 0., 0., 0., 0.])
        a[1,:] = array([ 1./18                  , 0.   ,   0.   , 0.                       , 0., 0., 0., 0., 0., 0., 0., 0.])
        a[2,:] = array([ 1./48                  , 1./16,   0.   , 0.                       , 0., 0., 0., 0., 0., 0., 0., 0.])
        a[3,:] = array([ 1./32                  , 0.   ,   3./32, 0.                       , 0., 0., 0., 0., 0., 0., 0., 0.])
        a[4,:] = array([ 5./16                  , 0.   , -75./64, 75./64                   , 0., 0., 0., 0., 0., 0., 0., 0.])
        a[5,:] = array([ 3./80                  , 0.   ,   0.   , 3./16                    , 3./20, 0., 0., 0., 0., 0., 0., 0.])
        a[6,:] = array([ 29443841./614563906    , 0.   ,   0.   , 77736538./692538347      , -28693883./1125000000, 23124283./1800000000, 0., 0., 0., 0., 0., 0.])
        a[7,:] = array([ 16016141./946692911    , 0.   ,   0.   , 61564180./158732637      , 22789713./633445777, 545815736./2771057229, -180193667./1043307555, 0., 0., 0., 0., 0.])
        a[8,:] = array([ 39632708./573591083    , 0.   ,   0.   , -433636366./683701615    , -421739975./2616292301, 100302831./723423059, 790204164./839813087, 800635310./3783071287, 0., 0., 0., 0.])
        a[9,:] = array([ 246121993./1340847787 , 0.   ,   0.   , -37695042795./15268766246, -309121744./1061227803, -12992083./490766935, 6005943493./2108947869, 393006217./1396673457, 123872331./1001029789, 0., 0., 0.])
        a[10,:] = array([ -1028468189./846180014, 0.   ,   0.   , 8478235783./508512852    , 1311729495./1432422823, -10304129995./1701304382, -48777925059./3047939560, 15336726248./1032824649, -45442868181./3398467696, 3065993473./597172653, 0., 0.])
        a[11,:] = array([ 185892177./718116043  , 0.   ,   0.   , -3185094517./667107341   , -477755414./1098053517, -703635378./230739211, 5731566787./1027545527, 5232866602./850066563, -4093664535./808688257, 3962137247./1805957418, 65686358./487910083, 0.])
        a[12,:] = array([ 403863854./491063109  , 0.   ,   0.   , -5068492393./434740067   , -411421997./543043805, 652783627./914296604, 11173962825./925320556, -13158990841./6184727034, 3936647629./1978049680, -160528059./685178525, 248638103./1413531060, 0.])
        #-------------------------------------------------------------------------------------
        b[:] = array([ 14005451./335480064, 0., 0., 0., 0., -59238493./1068277825, 181606767./758867731, 561292985./797845732, -1041891430./1371343529, 760417239./1151165299, 118820643./751138087, -528747749./2220607170, 1./4])
        bs[:] = array([ 13451932./455176623, 0., 0., 0., 0., -808719846./976000145, 1757004468./5645159321, 656045339./265891186, -3867574721./1518517206, 465885868./322736535, 53011238./667516719, 2./45, 0.])

        return a, b, bs, c, q

    else:

       print(" Error Butcher array: ")
       print( name, " is not available")

       return None


def Embedded_RK_U_n1(Function, Un, t, h, tag, name="RK65"): # This Embedded Runge Kutta computes U_{n+1} according to the selected scheme in ERK_selection
    a, b, b2, c, q = ERK_Selection(name)
    k = zeros([len(Un), len(b)]) # k_i is k[:,i]

    for i, ai in enumerate(a):

        S = zeros(len(Un))

        for j in range(i): # j€[0, i-1]

            S = S + ai[j]*k[:,j]

        k[:,i] = Function(Un + h*S, t+c[i]*h)

    if tag == "1":
        U_n1 = Un + h * matmul( b, k.transpose() ) # U_{n+1} = U_{n} + dt * \Sum_{i=0}^{i=s-1} (b_i * k_i); s == len(a)
        return U_n1, q

    else:
        U_n1_2 = Un + h * matmul( b2, k.transpose() )
        return U_n1_2






# %% Testeo de funciones



#

# def Absolute_Stability_Region(Temporal_Scheme, lst):

#     if Temporal_Scheme == "Euler":
#         x = linspace(-4,0,100)
#         y = linspace(-1.5,1.5,100)
#     elif Temporal_Scheme == "RK4":
#         x = linspace(-4,0.5,100)
#         y = linspace(-3,3,100)
#     elif Temporal_Scheme == "Inverse Euler":
#         x = linspace(-1,2.25,100)
#         y = linspace(-3,3,100)
#     else:
#         pass

#     x = linspace(-5,5,100)
#     y = linspace(-5,5,100)
#     N = size(x); M = size(y)
#     Z = zeros([N,M],dtype=complex)

#     for i in range(N):
#         for j in range(M):
#             Z[N-1-j,i] = complex(x[i],y[j])


#     Reg_Stability = absolute(array(Stability_Polynomial(Temporal_Scheme, Z)))
