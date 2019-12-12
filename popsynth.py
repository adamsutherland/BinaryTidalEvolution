#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:07:07 2019
Old version of pop synth. Use e_crit.py instead.
@author: adam
"""

import numpy as np

import imp
tides = imp.load_source('tides', '/Users/adam/Code/BinaryTidalEvolution/tides.py')
qs = imp.load_source('quicksilver', '/Users/adam/Code/quicksilver/quicksilver.py')


#q=ms/mp

P_b = 3*10**np.arange(0,1.1,.1)

e_b = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.3, 0.4, 0.5, 0.6]

m1 = np.ones(11)


q1 = np.linspace(0.08,0.94,8) 
q2 = np.linspace(0.95,1,2)

q = np.append(q1,q2)


#qs.sma(m1,m1*q,P_b)

# Things needed for run_dual()
# Name, m1, m2, abin, ebin, r1, r2, l1, l2,(Temp) 

f0 = open("../../Projects/tidal/popsynth/systems.txt","w+")
count = 0
for p in P_b:
    for q0 in q:
        m1 = 1.0
        m2 = m1*q0
        a = qs.sma(m1,m2,p)
        r1 = tides.R_Tout(m1)
        r2 = tides.R_Tout(m2)
        l1 = tides.L_Tout(m1)
        l2 = tides.L_Tout(m2)
        for e in e_b:
            f0.write(str(count).zfill(5)+", ")
            f0.write(str(m1)+", ")
            f0.write(str(m2)+", ")
            f0.write(str(a)+", ")
            f0.write(str(e)+", ")
            f0.write(str(r1)+", ")
            f0.write(str(r2)+", ")
            f0.write(str(l1)+", ")
            f0.write(str(l2)+"\n")
            count += 1

f0.close() 
   
def run_dual(row):
    fileexists = False
    #print "name = ",str(int(row['Name']))
    for s in runs:
        if str(int(row['Name'])) in s:
            fileexists = True
    if fileexists:
        print "skip"
    else:
        tmax = 5*10**9
        tout = 10**5
        dt = 10**4
        n = int(tmax/tout)
        tt = np.zeros(n)
        aa = np.zeros(n)
        ee = np.zeros(n)
        ww = np.zeros(n)
        
        print row['Name']
        m1,m2 = row["m1"], row["m2"]
        a0,e0 = row['abin'],row['ebin']
        t0 =0
        
        R1 = row["R1"]#/215.0
        Renv1, Menv1 = tides.R_M_env(m1,R1)
        L1 = row["L1"]
        Tau1 = 0.4311*((Menv1*Renv1*(R1-Renv1/2.))/(3.*L1))**(1./3.)
        MenvM1 = Menv1/m1
        R1 = R1/215.0
        
        R2 = row["R2"]#/215.0
        Renv2, Menv2 = tides.R_M_env(m2,R2)
        L2 = row["L2"]
        Tau2 = 0.4311*((Menv2*Renv2*(R2-Renv2/2.))/(3.*L2))**(1./3.)
        MenvM2 = Menv2/m2
        R2 = R2/215.0
        
        for x in xrange(n):
            t0, a0, e0, w0, f1, f2 = tides.rungeKuttaPSdual(t0, a0, e0, m1, m2, R1, R2, MenvM1, MenvM2, Tau1, Tau2, t0+ tout, dt)
            tt[x], aa[x], ee[x], ww[x] = t0, a0, e0, w0
        df = tides.pd.DataFrame(data={'t': tt, 'a': aa, 'e': ee, 'w': ww})
        df.to_pickle("../../Projects/tidal/popsynth/f30/"+str(int(row['Name'])).zfill(5)+".p")



systems = qs.pd.read_csv("../../Projects/tidal/popsynth/systems.txt", names=["Name","m1","m2","abin","ebin","R1","R2","L1","L2"])

systems = systems[systems.Name>=380]

#
#import matplotlib.pyplot as plt
#
#ms = np.arange(0.1,100,0.1)
#rs = tides.R_Tout(ms)
#ls = tides.L_Tout(ms)
#
#
#plt.figure()
#plt.loglog(ms,rs)
#plt.grid(True)
#plt.show()
#
#plt.figure()
#plt.loglog(ms,ls)
#plt.grid(True)
#plt.show()

ftide = "f30"
import glob as glob
runs = glob.glob("../../Projects/tidal/popsynth/"+ftide+"/*.p")


import multiprocessing as mp

numcpu = mp.cpu_count()
rows = [row for index, row in systems.iterrows()]

if len(rows) % numcpu != 0:
    batches = len(rows)/numcpu+1
    numcpu = len(rows)/batches+1
    print numcpu

pool = mp.Pool(processes=numcpu)
pool.map(run_dual, rows)








