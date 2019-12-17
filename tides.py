#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 00:09:52 2019

@author: adam
"""

import multiprocessing as mp
import pandas as pd
import numpy as np

# Eccentricity functions

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

def rtms(m):
    """Not used. Finds radius on main sequence."""
    a17, a18, a19, a20 = 1.4, 2.187715, 1.466440, 2.652091
    a22, a23, a24, a25 = 3.071048, 2.617890, 1.075567, 1.476246
    a21, a26, c1 = 1.47, 5.502535, -8.672073*10**-2
    #a21 = 1.47
    #a22 = 3.07
    #a20 = 2.652091*10**1
    #a19 = 1.466440*10**0 *a20
    #a18 = 2.187715*10**-1 *a20
    if m <= a17:
        r = (a18 + a19*m**a21)/(a20 + m**a22)
    #if m >=(a17+0.1):
    #    r = (c1*m**3 + a23*m**a26 + a24*m**(a26+1.5))/(a25 + m**5)
    return r
    #return 0.338
        
def rt(m,t):
    """Not used."""
    tms = 5*10**11
    tau = t/tms
    alpha_r = 8.4300*10**-2
    logs = alpha_r * tau

def L_Tout(m):
    """Finds luminosity on the MS according to Tout 1996"""
    return (0.39704170*m**5.5+8.52762600*m**11)/(0.00025546+m**3+5.43288900*m**5+5.56357900*m**7+0.78866060*m**8+0.00586685*m**9.5)

def R_Tout(m):
    """Finds radius on the MS according to Tout 1996"""
    r1 = (1.71535900*m**2.5+ 6.59778800*m**6.5+ 10.08855000*m**11+ 1.01249500*m**19+ 0.07490166*m**19.5)
    r2 = (0.01077422 + 3.08223400*m**2+ 17.84778000*m**8.5+ m**18.5+ 0.00022582*m**19.5)
    return r1/r2

# Reads table of Kepler binaries and planets
csv = pd.read_csv("kepler_data.csv",delim_whitespace=True)
csv = csv[(csv.Name != "47d")&(csv.Name != "47c")] # removes redundant planets
##csv = csv[(csv.Name != "64b")]



rg2=0.1 # radius of gyration
G = 4.0*np.pi**2 # gravity constant
#R = 1./215 # conversion between AU and stellar radius
fob = -1 # forward or backward evolution, backward -1

# tidal evolution equations with rotation
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

def ps_approx(e):
    return 1. + 6*e**2 + 3./8.*e**4 + 223./8.*e**6

# pseudo-synchronous ratio for ecc
def ps(e):
    ps1 = 1+15./2.*e**2+ 45./8.*e**4 + 5./16.*e**6
    ps2 = (1+3.*e**2+3./8.*e**4)*(1-e**2)**1.5
    return ps1/ps2

# equations for PS tides

ftide = 50    

def dadt2(a,e,w,k,Tau,m1,m2,R):
    if Tau == 0:
        return 0
    else:
        q=m2/m1
        return fob*-ftide*6*k/Tau*q*(1 + q)*(R/a)**8*a/(1 - (e)**2)**(15./2) * (f1(e) - (1 - (e)**2)**(3./2)*f2(e)* ps(e) )

def dedt2(a,e,w,k,Tau,m1,m2,R):
    if Tau == 0:
        return 0
    else:
        q=m2/m1
        return fob*-ftide*27*k/Tau*q*(1 + q)*(R/a)**8*e/(1 - (e)**2)**(13./2) * (f3(e) - 11./18*(1 - (e)**2)**(3./2)*f4(e)* ps(e))

def rungeKutta(t0, a0, e0, w0, m1, m2, R, Menv, Tau, t, h):
    """See rungeKuttaPSdual instead. This tracks spin as well"""
    # Count number of iterations using step size or 
    # step height h 
    n = (int)((t - t0)/h)  
    # Iterate for number of iterations 
    #a = a0
    Tau = 0.04533467056883912
    for i in range(1, n + 1): 
        "Apply Runge Kutta Formulas to find next value of y"
        Ptid = (abs((G*(m1+m2))**(1./2.)*(a0)**(-3./2)-w0))**-1
        k = 2.0/21.0 * min(1.0,(Ptid/(2*Tau))**2)*Menv
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

def rungeKuttaPS(t0, a0, e0, w0, m1, m2, R, MenvM, Tau, t, h):
    """See rungeKuttaPSdual instead. PS case, single star."""
    # Count number of iterations using step size or 
    # step height h 
    n = (int)((t - t0)/h)  
    # Iterate for number of iterations 
    #a = a0
    w0 = ps(e0)* ((G*(m1+m2))**(1./2.)*(a0)**(-3./2))
    for i in range(1, n + 1): 
        "Apply Runge Kutta Formulas to find next value of y"
        Ptid = (abs((G*(m1+m2))**(1./2.)*(a0)**(-3./2)-w0))**-1
        k = 2.0/21.0 * min(1.0,(Ptid/(2*Tau))**2) * MenvM
        #print Ptid, Tau
        #porb =  1/((G*(m1+m2))**(1./2.)*(a0)**(-3./2))
        #k = 2.0/21.0 * min(1.0,(porb/(2*Tau))**2) * MenvM
        #print porb/Tau
        #k = 2.0/21.0 * MenvM

        #q=m2/m1
        #print 50 * k/Tau#* q*(1+q), "  "
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


def rungeKuttaPSdual(t0, a0, e0, m1, m2, R1, R2, MenvM1, MenvM2, Tau1, Tau2, t, h):
    """RK4 for pseudo-synchronous case considering tides from both stars. t0,
    a0, e0 are the initial time, sma, and eccentricity. m1, m2, R1, R2 are the
    masses and radii of the stars. MenvM1, MenvM2, Tau1, Tau2 are the masses of 
    the envelopes and timescales. t is the final time and h is step size for 
    the integration"""
    # for two stars 
    n = (int)((t - t0)/h)  
    # Iterate for number of iterations 
    w0 = ps(e0)* ((G*(m1+m2))**(1./2.)*(a0)**(-3./2)) # initial spin
    for i in range(1, n + 1): 
        "Apply Runge Kutta Formulas to find next values"
        Ptid = (abs((G*(m1+m2))**(1./2.)*(a0)**(-3./2)-w0))**-1
        Porb =  1/((G*(m1+m2))**(1./2.)*(a0)**(-3./2)) # not actually used
        if Tau1 ==0:
            k1 =0
            f1 =0
        else:
            f1 = min(1.0,(Ptid/(2*Tau1))**2)
            #f1 = min(1.0,(Porb/(2*Tau1))**2)
            f1 = 1
            k1 = 2.0/21.0 * f1 * MenvM1
        if Tau2 ==0:
            k2 =0
            f2 =0
        else:
            f2 = min(1.0,(Ptid/(2*Tau2))**2)
            #f2 = min(1.0,(Porb/(2*Tau2))**2)
            f2 = 1
            k2 = 2.0/21.0 * f2 * MenvM2

        k1a = h * dadt2(a0,e0,w0,k1,Tau1,m1, m2,R1) + h * dadt2(a0,e0,w0,k2,Tau2,m2, m1,R2)
        k1e = h * dedt2(a0,e0,w0,k1,Tau1,m1, m2,R1) + h * dedt2(a0,e0,w0,k2,Tau2,m2, m1,R2)
        k2a = h * dadt2(a0 + 0.5*k1a,e0 + 0.5*k1e,w0,k1,Tau1,m1, m2,R1) + h * dadt2(a0 + 0.5*k1a,e0 + 0.5*k1e,w0,k2,Tau2,m2, m1,R2)
        k2e = h * dedt2(a0 + 0.5*k1a,e0 + 0.5*k1e,w0,k1,Tau1,m1, m2,R1) + h * dedt2(a0 + 0.5*k1a,e0 + 0.5*k1e,w0,k2,Tau2,m2, m1,R2)
        k3a = h * dadt2(a0 + 0.5*k2a,e0 + 0.5*k2e,w0,k1,Tau1,m1, m2,R1) + h * dadt2(a0 + 0.5*k2a,e0 + 0.5*k2e,w0,k2,Tau2,m2, m1,R2)
        k3e = h * dedt2(a0 + 0.5*k2a,e0 + 0.5*k2e,w0,k1,Tau1,m1, m2,R1) + h * dedt2(a0 + 0.5*k2a,e0 + 0.5*k2e,w0,k2,Tau2,m2, m1,R2)
        k4a = h * dadt2(a0 + k3a, e0 + k3e,w0,k1,Tau1,m1, m2,R1) + h * dadt2(a0 + k3a, e0 + k3e,w0,k2,Tau2,m2, m1,R2)
        k4e = h * dedt2(a0 + k3a, e0 + k3e,w0,k1,Tau1,m1, m2,R1) + h * dedt2(a0 + k3a, e0 + k3e,w0,k2,Tau2,m2, m1,R2)
  
        # Update next values
        a0 = a0 + (1.0 / 6.0)*(k1a + 2 * k2a + 2 * k3a + k4a)
        e0 = e0 + (1.0 / 6.0)*(k1e + 2 * k2e + 2 * k3e + k4e)
        w0 = ps(e0)* ((G*(m1+m2))**(1./2.)*(a0)**(-3./2))

  
        # Update next time
        t0 = t0 + h 
        # Also tracks f1 and f2 which are good diagnostics 
    return t0, a0, e0, w0, f1, f2

def run(row):
    """There are more comments on this function but run_dual should be used for
    all runs, regardless of secondary size. Probably deprecated"""
    tmax = 10**8 # the time for the integration
    tout = 10**5 # timestep between outputs
    dt = 10**4 # timestep for integration
    n = int(tmax/tout)
    # initialize arrays of the 
    tt = np.zeros(n)
    aa = np.zeros(n)
    ee = np.zeros(n)
    ww = np.zeros(n) # tracking spin even though it is set by PS
    #df2 = pd.read_pickle("../../Projects/tidal/forward/"+row['Name']+".p")
    print row['Name']
    m1,m2 = row["m1"], row["m2"]
    a0,e0,w0 = row['abin'],row['ebin'],(G*(m1+m2))**(1./2.)*row['abin']**(-3./2)*row['Pbin']/row['Prot']
    #a0,e0,w0 = df2["a"].iloc[-1], df2["e"].iloc[-1], df2["w"].iloc[-1]
    #a0,e0,w0 = row['abin'],row['ebin'],2.0*np.pi/(row['Prot']/365.0)
    #print 2.0*np.pi/(row['Prot']/365.0), (G*(m1+m2))**(1./2.)*row['abin']**(-3./2)*row['Pbin']/row['Prot']
    t0 =0
    # setting the mass and radius of the envelope.
    R = row["R1"]#/215.0 #should be in solar units
    if m1 <= 0.35:
        Renv = R
        Menv = m1
    if (m1 > 0.35) & (m1 < 1.25):
        Renv = 0.38*((1.25-m1)/0.9)**.3
        Menv = 0.35*((1.25-m1)/0.9)**2.0
    if m1 >= 1.25:
        Renv = 0.0
        Menv = 0.0
    L = (R)**2 *(row["T1"]/5772.0)**4
    #print row['Name'],m1, Menv, R, Renv
    Tau = 0.4311*((Menv*Renv*(R-Renv/2.))/(3.*L))**(1./3.)
    MenvM = Menv/m1
    R = R/215.0 # converting to AU
    for x in xrange(n):
        t0, a0, e0, w0 = rungeKuttaPS(t0, a0, e0, w0, m1,m2,R,MenvM,Tau,t0+ tout, dt)
        tt[x], aa[x], ee[x], ww[x] = t0, a0, e0, w0
    # data is saved as a pandas dataframe in the pickle format.
    df = pd.DataFrame(data={'t': tt, 'a': aa, 'e': ee, 'w': ww})
    df.to_pickle("../../Projects/tidal/"+row['Name']+".p")
    
def run_dual(row):
    tmax = 10**10
    tout = 10**5
    dt = 10**4
    n = int(tmax/tout)
    tt = np.zeros(n)
    aa = np.zeros(n)
    ee = np.zeros(n)
    ww = np.zeros(n)
    ff1 = np.zeros(n)
    ff2 = np.zeros(n)
    
    print row['Name']
    m1,m2 = row["m1"], row["m2"]
    #a0,e0,w0 = row['abin'],row['ebin'],(G*(m1+m2))**(1./2.)*row['abin']**(-3./2)*row['Pbin']/row['Prot']
    a0,e0 = row['abin'],row['ebin']
    #a0,e0,w0 = row['abin'],row['ebin'],2.0*np.pi/(row['Prot']/365.0)
    #print 2.0*np.pi/(row['Prot']/365.0), (G*(m1+m2))**(1./2.)*row['abin']**(-3./2)*row['Pbin']/row['Prot']
    t0 =0
    
    R1 = row["R1"]#/215.0
    Renv1, Menv1 = R_M_env(m1,R1)
    L1 = (R1)**2 *(row["T1"]/5772.0)**4
    Tau1 = 0.4311*((Menv1*Renv1*(R1-Renv1/2.))/(3.*L1))**(1./3.)
    MenvM1 = Menv1/m1
    R1 = R1/215.0
    
    R2 = row["R2"]#/215.0
    Renv2, Menv2 = R_M_env(m2,R2)
    L2 = (R2)**2 *(row["T2"]/5772.0)**4
    Tau2 = 0.4311*((Menv2*Renv2*(R2-Renv2/2.))/(3.*L2))**(1./3.)
    MenvM2 = Menv2/m2
    R2 = R2/215.0
    
    for x in xrange(n):
        t0, a0, e0, w0, f1, f2 = rungeKuttaPSdual(t0, a0, e0, m1, m2, R1, R2, MenvM1, MenvM2, Tau1, Tau2, t0+ tout, dt)
        tt[x], aa[x], ee[x], ww[x], ff1[x], ff2[x] = t0, a0, e0, w0, f1, f2
    df = pd.DataFrame(data={'t': tt, 'a': aa, 'e': ee, 'w': ww, 'f1': ff1, 'f2': ff2})
    df.to_pickle("../../Projects/tidal/"+str(ftide)+"/"+row['Name']+".p")


def R_M_env(m,R):
    """Mass and radius of convective envelope"""
    if m <= 0.35:
        Renv = R
        Menv = m
    if (m > 0.35) & (m < 1.25):
        Renv = 0.38*((1.25-m)/0.9)**.3
        Menv = 0.35*((1.25-m)/0.9)**2.0
    if m >= 1.25:
        Renv = 0.0
        Menv = 0.0
    return Renv, Menv

def run_dual_R_evo(row):
    """Accounts for radius evolution. Reads in stellar evolution track."""
    track = pd.read_csv("./tracks/"+row['Name']+".DAT",delim_whitespace=True)
    track["R"] = 10**track.LOG_R/6.9598e10
    track["L"] = 10**track.LOG_L
    trmin = track.AGE[track.R==track.R.min()].values[0] # time at minimum radius
    tzams = track.AGE[track.PHASE==5.0].values[0] # age at PHASE 5, ZAMS
    rzams = track.R[track.PHASE==5.0].values[0] # radius at ZAMS
    ttams = track.AGE[track.PHASE==8.0].values[0] # time of TAMS
    track_r=track[track.AGE>trmin] # removes preMS part of track
    
    current_age = np.interp(row["R1"],track_r.R,track_r.AGE)    
    #tmax = 10**10
    tmax = current_age - tzams
    print row['Name'], row["R1"], current_age, tmax, np.log10(tmax)
    tout = 10**5
    dt = 10**4
    n = int(tmax/tout)
    tt = np.zeros(n)
    aa = np.zeros(n)
    ee = np.zeros(n)
    ww = np.zeros(n)
    ff1 = np.zeros(n)
    ff2 = np.zeros(n)
    
    print row['Name']
    m1,m2 = row["m1"], row["m2"]
    #a0,e0,w0 = row['abin'],row['ebin'],(G*(m1+m2))**(1./2.)*row['abin']**(-3./2)*row['Pbin']/row['Prot']
    a0,e0 = row['abin'],row['ebin']
    #a0,e0,w0 = row['abin'],row['ebin'],2.0*np.pi/(row['Prot']/365.0)
    #print 2.0*np.pi/(row['Prot']/365.0), (G*(m1+m2))**(1./2.)*row['abin']**(-3./2)*row['Pbin']/row['Prot']
    t0 =0
    
    for x in xrange(n):
   
        #R1 = row["R1"]
        R1 = np.interp(current_age-t0,track.AGE,track.R)
        Renv1, Menv1 = R_M_env(m1,R1)
        Renv1 +=  R1 - rzams
        #tau = (current_age-t0-tzams)/(ttams-tzams)
        #Renv1 = Renv1 * (1-tau)**.25
        L1 = np.interp(current_age-t0,track.AGE,track.L)
        #L1 = (R1)**2 *(row["T1"]/5772.0)**4
        Tau1 = 0.4311*((Menv1*Renv1*(R1-Renv1/2.))/(3.*L1))**(1./3.)
        MenvM1 = Menv1/m1
        R1 = R1/215.0
        
        R2 = row["R2"]
        Renv2, Menv2 = R_M_env(m2,R2)
        L2 = (R2)**2 *(row["T2"]/5772.0)**4
        Tau2 = 0.4311*((Menv2*Renv2*(R2-Renv2/2.))/(3.*L2))**(1./3.)
        MenvM2 = Menv2/m2
        R2 = R2/215.0

        t0, a0, e0, w0, f1, f2 = rungeKuttaPSdual(t0, a0, e0, m1, m2, R1, R2, MenvM1, MenvM2, Tau1, Tau2, t0+ tout, dt)
        tt[x], aa[x], ee[x], ww[x], ff1[x], ff2[x] = t0, a0, e0, w0, f1, f2
    df = pd.DataFrame(data={'t': tt, 'a': aa, 'e': ee, 'w': ww, 'f1': ff1, 'f2': ff2})
    df.to_pickle("../../Projects/tidal/"+str(ftide)+"/"+row['Name']+"_evo.p")

import multiprocessing


numcpu = multiprocessing.cpu_count()
rows = [row for index, row in csv.iterrows()]


# uncomment to run all Kepler with constant radius
#
# #efficiently splits processes between cores
#if len(rows) % numcpu != 0:
#    batches = len(rows)/numcpu+1
#    numcpu = len(rows)/batches+1
#
# #multiprocessing for all 
#pool = mp.Pool(processes=numcpu)
#pool.map(run_dual, rows)


# #limit csv to just values with radius evolution:
#csv = csv[(csv.Name == "34b") |  (csv.Name == "38b") | (csv.Name == "47b") | (csv.Name == "KIC")]
#rows = [row for index, row in csv.iterrows()]
#
#pool = mp.Pool(processes=numcpu)
#pool.map(run_dual_R_evo, rows)


# Extra:
# to run just one at a time:
#run_dual(rows[3])
#run_dual_R_evo(rows[3])


#for index, row in csv.iterrows():
#    print row["Name"]




