#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 00:09:52 2019

@author: adam
"""

import multiprocessing as mp
import pandas as pd
import numpy as np




def f1(e2):
    return 1 + 31./2 *e2**2 + 255./8 *e2**4 + 185./16 *e2**6 + 25./64 *e2**8

def f2(e2):
    return 1 + 15./2 *e2**2 + 45./8 *e2**4 + 5./16 *e2**6

def f3(e2):
    return 1 + 15./4 *e2**2 + 15./8 *e2**4 + 5./64 *e2**6

def f4(e2):
    return 1 + 3./2 *e2**2 + 1./8 *e2**4

def f5(e2):
    return 1 + 3*e2**2 + 3./8 *e2**4


#def logRMS(t):
#    ar*tau + br*tau**10 + gamma*tau**40 + 

#def rtms(m):
#    if m <= a17:
#        r = (a18 + a19*m**a21)/(a20 + m**a22)
#    if m >=(a17+0.1):
#        r = (c1*m**3 + a23*m**a26 + a24*m**(a26+1.5))/(a25 + m**5)
#    return r
        


csv = pd.read_csv("kepler_data.csv",delim_whitespace=True)
csv = csv[(csv.Name != "47d")&(csv.Name != "47c")]

rg2=0.1
G = 4.0*np.pi**2
R = 1./215
fob = -1

def dadt(a,e,w,k,Tau,m1,m2,R):
    q=m2/m1
    return fob*-50*6*k/Tau*q*(1 + q)*(R/a)**8*a/(1 - (e)**2)**(15./2) * (f1(e) - (1 - (e)**2)**(3./2)*f2(e)* w/((G*(m1+m2))**(1./2.)*(a)**(-3./2)))

def dedt(a,e,w,k,Tau,m1,m2,R):
    q=m2/m1
    return fob*-50*27*k/Tau*q*(1 + q)*(R/a)**8*e/(1 - (e)**2)**(13./2) * (f3(e) - 11./18*(1 - (e)**2)**(3./2)*f4(e)* w/((G*(m1+m2))**(1./2.)*(a)**(-3./2)))

def dwdt(a,e,w,k,Tau,m1,m2,R):
    q=m2/m1
    return fob*1*3*k/Tau*q**2/rg2*(R/a)**6*(G*(m1+m2))**(1./2.)*(a)**(-3./2)/(1 - (e)**2)**(6.) * (f2(e) - (1 - (e)**2)**(3./2)*f5(e)* w/((G*(m1+m2))**(1./2.)*(a)**(-3./2)))


#eq tides

def ps(e):
    return 1. + 6*e**2 + 3./8.*e**4 + 223./8.*e**6

def dadt2(a,e,w,k,Tau,m1,m2,R):
    q=m2/m1
    return fob*-50*6*k/Tau*q*(1 + q)*(R/a)**8*a/(1 - (e)**2)**(15./2) * (f1(e) - (1 - (e)**2)**(3./2)*f2(e)* ps(e) )

def dedt2(a,e,w,k,Tau,m1,m2,R):
    q=m2/m1
    return fob*-50*27*k/Tau*q*(1 + q)*(R/a)**8*e/(1 - (e)**2)**(13./2) * (f3(e) - 11./18*(1 - (e)**2)**(3./2)*f4(e)* ps(e))


def rungeKutta(t0, a0, e0, w0, m1, m2, R, t, h): 
    # Count number of iterations using step size or 
    # step height h 
    n = (int)((t - t0)/h)  
    # Iterate for number of iterations 
    #a = a0
    Tau = 0.04533467056883912
    R = 1./215. * m1**(3./7.)
    for i in range(1, n + 1): 
        "Apply Runge Kutta Formulas to find next value of y"
        Ptid = (abs((a0)**(-3./2)-w0))**-1
        k = 2.0/21.0 * min(1.0,(Ptid/(2*Tau))**2)
        k1a = h * dadt(a0,e0,w0,k,Tau,m1, m2,R) 
        k1e = h * dedt(a0,e0,w0,k,Tau,m1, m2,R) 
        k1w = h * dwdt(a0,e0,w0,k,Tau,m1, m2,R)
        k2a = h * dadt(a0 + 0.5*k1a,e0 + 0.5*k1e,w0 + 0.5*k1w,k,Tau,m1, m2,R)
        k2e = h * dedt(a0 + 0.5*k1a,e0 + 0.5*k1e,w0 + 0.5*k1w,k,Tau,m1, m2,R)
        k2w = h * dwdt(a0 + 0.5*k1a,e0 + 0.5*k1e,w0 + 0.5*k1w,k,Tau,m1, m2,R)
        k3a = h * dadt(a0 + 0.5*k2a,e0 + 0.5*k2e,w0 + 0.5*k2w,k,Tau,m1, m2,R)
        k3e = h * dedt(a0 + 0.5*k2a,e0 + 0.5*k2e,w0 + 0.5*k2w,k,Tau,m1, m2,R)
        k3w = h * dwdt(a0 + 0.5*k2a,e0 + 0.5*k2e,w0 + 0.5*k2w,k,Tau,m1, m2,R)
        k4a = h * dadt(a0 + k3a, e0 + k3e, w0 + k3w,k,Tau,m1, m2,R)
        k4e = h * dedt(a0 + k3a, e0 + k3e, w0 + k3w,k,Tau,m1, m2,R)
        k4w = h * dwdt(a0 + k3a, e0 + k3e, w0 + k3w,k,Tau,m1, m2,R)
  
        # Update next value of y 
        a0 = a0 + (1.0 / 6.0)*(k1a + 2 * k2a + 2 * k3a + k4a)
        e0 = e0 + (1.0 / 6.0)*(k1e + 2 * k2e + 2 * k3e + k4e)
        w0 = w0 + (1.0 / 6.0)*(k1w + 2 * k2w + 2 * k3w + k4w)
  
        # Update next value of x 
        t0 = t0 + h 
    return t0, a0, e0, w0

def rungeKuttaPS(t0, a0, e0, w0, m1, m2, R, t, h): 
    # Count number of iterations using step size or 
    # step height h 
    n = (int)((t - t0)/h)  
    # Iterate for number of iterations 
    #a = a0
    Tau = 0.04533467056883912
    R = 1./215. * m1**(3./7.)
    w0 = ps(e0)* ((G*(m1+m2))**(1./2.)*(a0)**(-3./2))
    for i in range(1, n + 1): 
        "Apply Runge Kutta Formulas to find next value of y"
        Ptid = (abs((a0)**(-3./2)-w0))**-1
        k = 2.0/21.0 * min(1.0,(Ptid/(2*Tau))**2)
        k1a = h * dadt2(a0,e0,w0,k,Tau,m1, m2,R) 
        k1e = h * dedt2(a0,e0,w0,k,Tau,m1, m2,R) 
        k2a = h * dadt2(a0 + 0.5*k1a,e0 + 0.5*k1e,w0,k,Tau,m1, m2,R)
        k2e = h * dedt2(a0 + 0.5*k1a,e0 + 0.5*k1e,w0,k,Tau,m1, m2,R)
        k3a = h * dadt2(a0 + 0.5*k2a,e0 + 0.5*k2e,w0,k,Tau,m1, m2,R)
        k3e = h * dedt2(a0 + 0.5*k2a,e0 + 0.5*k2e,w0,k,Tau,m1, m2,R)
        k4a = h * dadt2(a0 + k3a, e0 + k3e,w0,k,Tau,m1, m2,R)
        k4e = h * dedt2(a0 + k3a, e0 + k3e,w0,k,Tau,m1, m2,R)
  
        # Update next value of y 
        a0 = a0 + (1.0 / 6.0)*(k1a + 2 * k2a + 2 * k3a + k4a)
        e0 = e0 + (1.0 / 6.0)*(k1e + 2 * k2e + 2 * k3e + k4e)
        w0 = ps(e0)* ((G*(m1+m2))**(1./2.)*(a0)**(-3./2))

  
        # Update next value of x 
        t0 = t0 + h 
    return t0, a0, e0, w0


def run(row):
    tmax = 10**10
    tout = 10**5
    dt = 10**4
    n = int(tmax/tout)
    tt = np.zeros(n)
    aa = np.zeros(n)
    ee = np.zeros(n)
    ww = np.zeros(n)
    print row['Name']
    m1,m2 = row["m1"], row["m2"]
    a0,e0,w0 = row['abin'],row['ebin'],(G*(m1+m2))**(1./2.)*row['abin']**(-3./2)*row['Pbin']/row['Prot']
    #a0,e0,w0 = row['abin'],row['ebin'],2.0*np.pi/(row['Prot']/365.0)
    #print 2.0*np.pi/(row['Prot']/365.0), (G*(m1+m2))**(1./2.)*row['abin']**(-3./2)*row['Pbin']/row['Prot']
    t0 =0
    R = row["R1"]
    for x in xrange(n):
        t0, a0, e0, w0 = rungeKuttaPS(t0, a0, e0, w0, m1,m2,R,t0+ tout, dt)
        tt[x], aa[x], ee[x], ww[x] = t0, a0, e0, w0
    df = pd.DataFrame(data={'t': tt, 'a': aa, 'e': ee, 'w': ww})
    df.to_pickle("../../Projects/tidal/"+row['Name']+".p")


import multiprocessing

numcpu = multiprocessing.cpu_count()
rows = [row for index, row in csv.iterrows()]

if len(rows) % numcpu != 0:
    batches = len(rows)/numcpu+1
    numcpu = len(rows)/batches+1

pool = mp.Pool(processes=numcpu)
pool.map(run, rows)




#for index, row in csv.iterrows():
#    print row["Name"]




