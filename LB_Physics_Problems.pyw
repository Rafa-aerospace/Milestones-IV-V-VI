# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 16:22:30 2022

@author: Rafael Rivero de Nicolás
"""

from numpy import array, reshape, zeros, sqrt
from numpy.linalg import norm

def N_Bodies_Function(U, t):

    if int(len(U))%6 == 0:
        Nb, Nc = (int(len(U)/6), 3)
    else:
        print("Error; U must has a dimension of 6*N, with N an integer")

    Us = reshape(U, (Nb, Nc*2))

    r = Us[:,0:Nc]
    # print(r)

    v = Us[:,Nc:]
    # print(v)

    F = zeros(len(U))                       # Vector that will be F(U) and the end of the function
    F_pointer = reshape( F, (Nb, Nc*2) )

    drdt = F_pointer[:,0:Nc]
    dvdt = F_pointer[:,Nc:]

    drdt[:,:] = v[:,:]

    for i in range(Nb):

        # print("i =", i)

        dvdt[i,:] = 0

        for j in range(Nb):

            if j != i:

                d = r[j,:]-r[i,:]
                # print(d)

                dvdt[i,:] = dvdt[i,:] + (d)/norm(d)**3
                # print(dvdt)

    return F

def Problem_Assignment(problem,Physics_Problems_available):

    if problem == Physics_Problems_available[0]:

        return Kepler_Orbits_2N

    elif problem == Physics_Problems_available[1]:

        return Undamped_Armonic_Oscilator

    else:
        print("Introduce a valid problem equation to solve\n\t", Physics_Problems_available )
        return "ERROR"

def Kepler_Orbits_2N(X, t):
    '''
    This function only depends on the physics of the problem, it musts be an input argument

    Parameters
    ----------
    X : Array
        State vector of the system in instant t.
    t : Float
        Time instant in which F is being evaluated.

    Returns
    -------
    Array
        First derivate of the tate vector. dU/dt = F(U,t).

    '''
    return array([X[2], X[3], -X[0]/(X[0]**2 + X[1]**2)**(3/2), -X[1]/(X[0]**2 + X[1]**2)**(3/2)])


def Undamped_Armonic_Oscilator(X, t):
    '''
    This function only depends on the physics of the problem, it musts be an input argument

    Parameters
    ----------
    X : Array
        State vector of the system in instant t.
    t : Float
        Time instant in which F is being evaluated.

    Returns
    -------
    Array
        First derivate of the tate vector. dU/dt = F(U,t).

    '''
    return array([X[1], -X[0]])


# %% Milestone 6

def Autonomous_F(U):

    R3BP_no_t =  R3BP_Earth_Moon(U, t=0)

    return R3BP_no_t


def R3BP_Earth_Moon(U, t): # Restricted 3 body problem

    mu = 0.0121505856

    # F = zeros( len( U ) )

    x  = U[0];   y = U[1];   z = U[2];
    vx = U[3];  vy = U[4];  vz = U[5];

    d = sqrt( (x+mu)**2 +y**2 + z**2 )
    r = sqrt( (x-1+mu)**2 + y**2 + z*2 )


    ax = x + 2 * vy - (1-mu) * ( x + mu )/d**3 - mu*(x-1+mu)/r**3
    ay = y - 2 * vx - (1-mu) * y/d**3 - mu * y/r**3
    az = - (1-mu)*z/d**3 - mu*z/r**3

    return array( [ vx, vy, vz, ax, ay, az ] )
