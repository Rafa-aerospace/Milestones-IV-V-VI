# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 21:03:25 2022

@author: Rafael Rivero de Nicolás
"""

from numpy import zeros, linalg, matmul #, array
# from LB_Temporal_Schemes import RK4, Leap_Frog, Euler, Inverse__Euler, Crank__Nicolson



# %% Useful Computing Functions

def Newton_Raphson(Eq, x_i):

    eps = 1; iteration = 1

    if (type(x_i)==int) or ((type(x_i)==float)) or (type(x_i)==complex):

        while eps>1E-10 and iteration<1E2:

            Jacobian = Numeric_Jacobian(F = Eq, x = x_i)

            x_f = x_i - Eq(x_i)/Jacobian

            iteration = iteration + 1

            eps = linalg.norm(x_f - x_i)

            x_i = x_f

    else:

        while eps>1E-10 and iteration<1E2:

            Jacobian = Numeric_Jacobian(F = Eq, x = x_i)

            x_f = x_i - matmul( linalg.inv( Jacobian ), Eq(x_i) )

            iteration = iteration + 1

            eps = linalg.norm(x_f - x_i)

            x_i = x_f


    return x_f


def Numeric_Jacobian(F, x):
    '''
    Parameters
    ----------
    F : Function
        Vectorial function depending on x that is wanted to be solved.
    x : Array of floats
        Variable of F.

    Returns
    -------
    Jacobian : Matrix
        This matrix allows to compute the derivate of F.

    '''

    try:

        Jacobian = zeros([len(x), len(F(x))])

        for column in range(len(Jacobian[0,:])):

            dx = zeros(len(x))

            dx[column] = 1E-10

            Jacobian[:,column] = ( F(x+dx)  - F(x-dx) ) / linalg.norm( 2 * dx ) # Second order finite diferences aproximation

    except: # Path that the program will follow if x and F are not vectorial

        Jacobian = 0

        dx = 1E-10

        Jacobian = ( F(x+dx)  - F(x-dx) ) /( 2 * dx ) # Second order finite diferences aproximation

    return Jacobian
