#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:34:29 2019

@author: adam
"""


import numpy as np
import pandas as pd


import imp
qs = imp.load_source('quicksilver', '/Users/adam/Code/quicksilver/quicksilver.py')
#reso = imp.load_source('resonances', '/home/adam/Code/analysis/resonances/reso.py')

csv = pd.read_csv("kepler_data.csv",delim_whitespace=True)
csv = csv[(csv.Name != "47d")&(csv.Name != "47c")]
csv = csv[(csv.Name == "34b")]
#csv = csv[csv.index < 5]
#csv = csv[(csv.Name != "16b")]
#csv = csv[(csv.Name != "35b")]

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

def check(m1,m2,d,ei,eo,ap):
    n1 = qs.mean_mo(m1,m2,d)
    n = int(n1/qs.mod_mean_mo(m1,m2,d,ap))
    width = qs.sigma_mard(m1,m2,0.,eo,ei,n)
    a = qs.mod_a_from_n(m1,m2,d,n1/(n))
    wdot = qs.mod_mean_mo(m1,m2,d,a)-qs.mod_epic(m1,m2,d,a)
    wdot *=(1+1.5*ei**2)
    n2 = (n1+(n-1)*wdot)/(n)
    #print qs.mod_a_from_n(m1,m2,d,n1/(width+n1/n2))
    #print qs.mod_a_from_n(m1,m2,d,n1/(-width+n1/n2))
    if ap > qs.mod_a_from_n(m1,m2,d,n1/(-width+n1/n2)):
        if ap < qs.mod_a_from_n(m1,m2,d,n1/(width+n1/n2)):
            inside =  n
        else:
            inside = 0
    else:
        inside = 0
    if inside < 1:
        n +=1
        width = qs.sigma_mard(m1,m2,0.,eo,ei,n)
        a = qs.mod_a_from_n(m1,m2,d,n1/(n))
        wdot = qs.mod_mean_mo(m1,m2,d,a)-qs.mod_epic(m1,m2,d,a)
        wdot *=(1+1.5*ei**2)
        n2 = (n1+(n-1)*wdot)/(n)
        #print qs.mod_a_from_n(m1,m2,d,n1/(width+n1/n2))
        #print qs.mod_a_from_n(m1,m2,d,n1/(-width+n1/n2))
        if ap > qs.mod_a_from_n(m1,m2,d,n1/(-width+n1/n2)):
            if ap < qs.mod_a_from_n(m1,m2,d,n1/(width+n1/n2)):
                inside =  n
            else:
                inside = 0
        else:
            inside = 0
    return inside#, a

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

ftide = "f30"


baseline = 1.5

def check_row(row):
    print row["Name"]
    df = pd.read_pickle("/home/adam/Projects/tidal/"+ftide+"/"+row['Name']+"_e.p")
    chaos = np.array([])
    for index2, row2 in df.iterrows():
        #ep = row2["e"]*row2["a"]/row["ap"]*(row["m1"]-row["m2"])/(row["m1"]+row["m2"])/0.4115
        ep = row["ep"]
        chaos = np.append(chaos,check(row["m1"],row["m2"],row2["a"],row2["e"],ep,row["ap"]))
    df["chaos"] = chaos
    print chaos.max()
    df.to_pickle("/home/adam/Projects/tidal/"+ftide+"/"+row['Name']+"_e.p")
    
def check_p(row):
    Name = str(int(row['Name'])).zfill(5)
    print Name
    df = pd.read_pickle("../../Projects/tidal/popsynth/"+ftide+"/"+Name+".p")
    chaos = np.array([])
    for index2, row2 in df.iterrows():
        ap = qs.sma(row["m1"],row["m2"],365*baseline)
        ep = row2["e"]*row2["a"]/ap*(row["m1"]-row["m2"])/(row["m1"]+row["m2"])/0.4115
        #ep = row["ep"]
        chaos = np.append(chaos,check_fast(row["m1"],row["m2"],row2["a"],row2["e"],ep,ap))
    df["chaos"] = chaos
    print chaos.max()
    df.to_pickle("../../Projects/tidal/popsynth/"+ftide+"/"+Name+".p")
    
def check_p_fast(row):
    Name = str(int(row['Name'])).zfill(5)
    print Name
    df = pd.read_pickle("../../Projects/tidal/popsynth/"+ftide+"/"+Name+".p")
    chaos = np.array([])
    check1 = 0
    overlap = False
    for index2, row2 in df.iterrows():
        m1, m2 = row["m1"],row["m2"]
        a, e = row2["a"],row2["e"]
        ap = qs.sma(m1, m2,365*baseline)
        #ep = e*a/ap*(m1-m2)/(m1+m2)/0.4115
        ep = 0.05
        if overlap:
            check0 = -1
        else:
            check0 = check_fast(m1,m2,a,e,ep,ap)
            if check1-check0 == 1.0:
                print check0, chaos[-1]
                overlap = True
        chaos = np.append(chaos,check0)
        check1 = check0
    df["chaos_05"] = chaos
    print chaos.max()
    df.to_pickle("../../Projects/tidal/popsynth/"+ftide+"/"+Name+".p")

import multiprocessing as mp

systems = qs.pd.read_csv("../../Projects/tidal/popsynth/systems.txt", names=["Name","m1","m2","abin","ebin","R1","R2","L1","L2"])

#systems = systems[systems.Name>406]
#systems = systems[systems.Name<408]


numcpu = mp.cpu_count()
rows = [row for index, row in systems.iterrows()]

if len(rows) % numcpu != 0:
    batches = len(rows)/numcpu+1
    numcpu = len(rows)/batches+1
pool = mp.Pool(processes=numcpu)
pool.map(check_p_fast, rows)

#for row in rows:
#    Name = str(int(row['Name'])).zfill(5)
#    print Name
#    qs.tic()
#    df = pd.read_pickle("../../Projects/tidal/popsynth/"+ftide+"/"+Name+".p")
#    chaos = np.array([])
#    for index2, row2 in df.iterrows():
#        m1, m2 = row["m1"],row["m2"]
#        a, e = row2["a"],row2["e"]
#        ap = qs.sma(m1, m2,365*baseline)
#        if a*hwac(m2/(m1 + m2),e) > ap:
#            check0 = -1
#        else:
#            ep = e*a/ap*(m1-m2)/(m1+m2)/0.4115
#            #ep = 0.04
#            check0 = check_fast(m1,m2,a,e,ep,ap)
#        chaos = np.append(chaos,check0)
#    df["chaos"] = chaos
#    print chaos.max()
#    df.to_pickle("../../Projects/tidal/popsynth/"+ftide+"/"+Name+".p")
#    qs.toc()







