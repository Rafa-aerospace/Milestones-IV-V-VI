# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 19:36:25 2022

@author: Rafael Rivero de Nicolás
"""

from numpy import zeros
import LB_Temporal_Schemes as ts

from numpy.linalg import norm


def Cauchy_Problem_V2(F, U_0, time_domain, Temporal_scheme=ts.RK4):

    print( 'Temporal Scheme used:: ' + Temporal_scheme.__name__ )

    t = 0.; U_n1 = zeros(len(U_0))

    U = zeros([len(U_0), len(time_domain)])

    U[:,0] = U_0

    if Temporal_scheme.__name__ == 'Leap_Frog':

        dt = time_domain[1] - time_domain[0]

        t = t + dt

        U[:,1] = ts.Euler(U[:,0], t, dt, F)
        # U[:,1] = array([1, 1]) # Invented value to appreciate "soluciones espúreas" of the Leap-Frog Scheme

        for i in range(1,len(time_domain)-1):

            dt = time_domain[i+1] - time_domain[i]

            t = t + dt

            U[:,i+1] = ts.Leap_Frog(U[:,i], U[:,i-1], t, dt, F)

        return U

    else:

        # try:

            for i in range(0, len(time_domain)-1):

                dt = time_domain[i+1] - time_domain[i]

                t = t + dt

                X = U[:,i]

                U_n1 = Temporal_scheme(X, t, dt, F)

                U[:,i+1] = U_n1

            return U

        # except:
            # print("Something happend during "+Temporal_scheme.__name__+" iterative aplication.")


def Cauchy_Problem(F, U_0, time_domain, Temporal_scheme='RK4'):

    print( 'Temporal Scheme used:: ' + Temporal_scheme )

    t = 0.; U_n1 = zeros(len(U_0))

    U = zeros([len(U_0), len(time_domain)])

    U[:,0] = U_0

    if Temporal_scheme == 'Leap Frog':

        dt = time_domain[1] - time_domain[0]

        t = t + dt

        U[:,1] = ts.Euler(U[:,0], t, dt, F)

        for i in range(1,len(time_domain)-1):

            dt = time_domain[i+1] - time_domain[i]

            t = t + dt

            U[:,i+1] = ts.Leap_Frog(U[:,i], U[:,i-1], t, dt, F)


    else:

        if Temporal_scheme == 'RK4':

            Numeric_Scheme = ts.RK4

        elif Temporal_scheme == 'Euler':

            Numeric_Scheme = ts.Euler

        elif Temporal_scheme == 'Crank-Nicolson':

            Numeric_Scheme = ts.Crank__Nicolson

        elif Temporal_scheme == 'Inverse Euler':

            Numeric_Scheme = ts.Inverse__Euler

        else:
            print('Introduce a valid Temporal scheme::\n\tEuler\n\tRK4\n\tCrank-Nicolson\n\tInverse Euler'+
                  '\n\n NUMERIC SCHEME USED: RUNKE-KUTTA 4')
            Numeric_Scheme = ts.RK4

        for i in range(0, len(time_domain)-1):

            dt = time_domain[i+1] - time_domain[i]

            t = t + dt

            X = U[:,i]

            U_n1 = Numeric_Scheme(X, t, dt, F)

            U[:,i+1] = U_n1

    return U


def Embedded_RK_Application(U_0, F, time_domain, name, tolerance, dt_min): # Global Embedded Runge Kutta application

    dt_min_reached = 0

    U = zeros( [len(U_0), len(time_domain)] )
    # print(U)

    U[:,0] = U_0
    # print(U)

    for i, t in enumerate(time_domain[1:]): # t = time_domain[i+1]

        dt = t-time_domain[i]
        # print(dt)

        U_n1, dt_min_reached = ERK_Temporal_Application( U[:,i], t, dt, F, dt_min_reached, tolerance, dt_min, name)

        U[:,i+1] = U_n1
        # print(U)

    return U, dt_min_reached



def Step_size_estimation(err, q, tolerance, h): # If err>tolerance, this function estimates the appropiate dt to apply

    return h * (tolerance/err)**(1./(q+1)) # Estimation of the new grid that would satisfy error criteria


def ERK_Temporal_Application(Un, t, dt, F, dt_min_reached, tolerance, dt_min, name): # Embedded Runge Kutta applicated to compute U in all the time_domain

    U_n1, q = ts.Embedded_RK_U_n1(F, Un, t, dt, tag="1", name = name)
    U_n1_2 = ts.Embedded_RK_U_n1(F, Un, t, dt, tag="2", name = name)

    err_estimation = norm( U_n1 - U_n1_2 ) # Truncation error estimation
    # print(err_estimation)

    if err_estimation >= tolerance:

        h = max ( [dt_min, Step_size_estimation(err_estimation, min(q), tolerance, dt)] )

        if h <= dt_min:

            dt_min_reached = dt_min_reached + 1

        N_grid = int(dt/h) + 1 # Number of intermediate steps betwwen t and t+dt

        h = dt/N_grid # h will be smaller than the h that satisfies tolerance, so this is conservative

        for k in range(N_grid):

            t1 = t + k*(dt)/N_grid

            # print(t1, h)

            if k == 0:

                U_n1, q = ts.Embedded_RK_U_n1(F, Un, t1, h, tag="1")

            else:

                U_n1, q = ts.Embedded_RK_U_n1(F, U_n1, t1, h, tag="1")


        return U_n1, dt_min_reached

    else:

        return U_n1, dt_min_reached

