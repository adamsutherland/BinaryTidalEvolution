#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:48:13 2019

@author: adam
"""

import numpy as np
import pandas as pd
import imp
tides = imp.load_source('tides', 'tides.py')
#qs = imp.load_source('quicksilver', '/Users/adam/Code/quicksilver/quicksilver.py')
qs = imp.load_source('quicksilver', '/home/adam/Code/analysis/quicksilver/quicksilver.py')


import multiprocessing as mp

numcpu = mp.cpu_count()

def run_dual(row):
    fileexists = False
    #print "name = ",str(int(row['Name']))
    #for s in runs:
    #    if str(int(row['Name'])) in s:
    #        fileexists = True
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
        
        print row['ebin']
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
        df.to_pickle("../../Projects/tidal/popsynth/e_crit/"+str(int(row['Name'])).zfill(5)+".p")






def mmm_factors(mp,ms,d):
    f1 = qs.G*45*mp*ms*(mp**2-mp*ms+ms**2)*d**4
    f2 = qs.G*48*mp*ms*(mp+ms)**2*d**2
    f3 = qs.G*64*(mp+ms)**4
    f4 = 64*(mp+ms)**3
    return f1, f2, f3, f4

def mmm_fast(f1,f2,f3,f4,a):
    n2 = (f1 + f2*a*a + f3*a*a*a*a)/(f4 * a*a*a*a*a*a*a)
    return n2**.5

def mod_a_from_n(mp, ms, d, n,f1,f2,f3,f4):
    a=qs.sma(mp,ms,2*np.pi/n)/2
    for x in xrange(16):
        #print x, a
        r=0.0
        count = 0
        while (r < 1):
            count +=1
            a = a + 10**-x
            r =  n/mmm_fast(f1,f2,f3,f4,a)
            if count > 100:
                break
        a=a-10**-x
    return a

def hwac(e,mu):
    return 1.60+5.10*e-2.22*e**2+4.12*mu-4.27*e*mu-5.09*mu**2+4.61*e**2*mu**2


def check_fast(m1,m2,d,ei,eo,ap):
    f1, f2, f3, f4 = mmm_factors(m1,m2,d)
    n1 = qs.mean_mo(m1,m2,d)
    n = int(n1/mmm_fast(f1,f2,f3,f4,ap))
    if n>40:
        inside = 0
    elif n<4:
        inside = 0
    else:
        width = qs.sigma_mard(m1,m2,0.,eo,ei,n)
        a = mod_a_from_n(m1,m2,d,n1/(n),f1,f2,f3,f4)
        wdot = mmm_fast(f1,f2,f3,f4,a)-qs.mod_epic(m1,m2,d,a)
        wdot *=(1+1.5*ei**2)
        n2 = (n1+(n-1)*wdot)/(n)
    
        if ap > mod_a_from_n(m1,m2,d,n1/(-width+n1/n2),f1,f2,f3,f4):
            if ap < mod_a_from_n(m1,m2,d,n1/(width+n1/n2),f1,f2,f3,f4):
                inside =  n
            else:
                inside = 0
        else:
            inside = 0
        if inside < 1:
            n +=1
            width = qs.sigma_mard(m1,m2,0.,eo,ei,n)
            a = mod_a_from_n(m1,m2,d,n1/(n),f1,f2,f3,f4)
            wdot = mmm_fast(f1,f2,f3,f4,a)-qs.mod_epic(m1,m2,d,a)
            wdot *=(1+1.5*ei**2)
            n2 = (n1+(n-1)*wdot)/(n)
    
            if ap > mod_a_from_n(m1,m2,d,n1/(-width+n1/n2),f1,f2,f3,f4):
                if ap < mod_a_from_n(m1,m2,d,n1/(width+n1/n2),f1,f2,f3,f4):
                    inside =  n
                else:
                    inside = 0
            else:
                inside = 0
    return inside#, a


def check_p_fast(row):
    Name = str(int(row['Name'])).zfill(5)
    print Name
    df = pd.read_pickle("../../Projects/tidal/popsynth/e_crit/"+Name+".p")
    chaos = np.array([])
    check1 = 0
    overlap = False
    baseline = 1.5
    if len(df[pd.isnull(df.a)])>0:
        overlap = True
    for index2, row2 in df.iterrows():
        m1, m2 = row["m1"],row["m2"]
        a, e = row2["a"],row2["e"]
        ap = qs.sma(m1, m2,365*baseline)
        es = e*a/ap*(m1-m2)/(m1+m2)/0.4115
        ef = 0.03
        ep = ef+es
        if a>ap:
            check0 = -1
        else:
            if overlap:
                check0 = -1
            else:
                check0 = check_fast(m1,m2,a,e,ep,ap)
                if check1-check0 == 1.0:
                    #print check0, chaos[-1]
                    overlap = True
        chaos = np.append(chaos,check0)
        check1 = check0
    df["chaos"] = chaos
    #name =  int(Name)
    #e0 = systems[systems.Name==row["Name"]].ebin[1]
    #print e0, chaos.max()
    df.to_pickle("../../Projects/tidal/popsynth/e_crit/"+Name+".p")


# Start #

#popsynth = pd.read_csv("../../Projects/tidal/popsynth/e_crit/popsynthresults.txt",names = ["Name","m1","m2","Pbin","ebin","cmax","cmin"])
#popsynth = popsynth[popsynth.m2=]

emin, emax = 0.01,.7

P_b = 7.0
P_b = 30

pbs = 3*10**np.arange(0,1.1,.05)
#pbs = pbs[-1:]
#pbs = [3.0]
q = [0.2,0.6,1.0]
#bases = [1.0,1.5]
#for baseline in bases:
for q0 in q:
    for P_b in pbs:
        emin, emax = 0.01,.7
    
        #q0 = 1.0
        
        m1 = 1.0
        m2 = m1*q0
        a = qs.sma(m1,m2,P_b)
        r1 = tides.R_Tout(m1)
        r2 = tides.R_Tout(m2)
        l1 = tides.L_Tout(m1)
        l2 = tides.L_Tout(m2)
        count = 0
        
        f0 = open("../../Projects/tidal/popsynth/e_crit/results.txt","w+")
        f0.close()
        
        for x in xrange(1):
            print x
            e_b = np.linspace(emin,emax,numcpu)
            
            f0 = open("../../Projects/tidal/popsynth/e_crit/systems.txt","w+")
            
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
        
            systems = qs.pd.read_csv("../../Projects/tidal/popsynth/e_crit/systems.txt", names=["Name","m1","m2","abin","ebin","R1","R2","L1","L2"])
            rows = [row for index, row in systems.iterrows()]
            
            pool = mp.Pool(processes=numcpu)
            pool.map(run_dual, rows)
            
            #baseline = 1.5
            
            pool = mp.Pool(processes=numcpu)
            pool.map(check_p_fast, rows)
            
            f0 = open("../../Projects/tidal/popsynth/e_crit/results.txt","a+")
            
            for name in systems.Name:
                Name = str(int(name)).zfill(5)
                df = pd.read_pickle("../../Projects/tidal/popsynth/e_crit/"+Name+".p")
                name = name - count+8
                f0.write(Name+", "+str(df.e.min())+", "+str(df.chaos.max())+", "+str(df.chaos.min())+"\n" )
                
            f0.close() 
            
            results = pd.read_csv("../../Projects/tidal/popsynth/e_crit/results.txt",names=["Name","ebin","cmax","cmin"])
            
            if len(results[results.cmin==-1])==0:
                ecrit = 1.0
                break
            elif len(results[results.cmin!=-1])==0:
                ecrit = 0.0
                break
            else:
                emin = results[results.cmin!=-1].ebin.max()
                emax = results[results.cmin==-1].ebin.min()
                ecrit = (emax+emin)/2
        
        
        f0 = open("../../Projects/tidal/popsynth/e_crit/meta_results_15.txt","a+")
        f0.write(str(q0)+", "+str(P_b)+", "+str(ecrit)+"\n")
        f0.close() 



