#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 00:01:18 2019

@author: adam
"""

import pandas as pd
import matplotlib.pyplot as plt
import imp
qs = imp.load_source('quicksilver', '/Users/adam/Code/quicksilver/quicksilver.py')

def hwac(e,mu):
    return 1.60+5.10*e-2.22*e**2+4.12*mu-4.27*e*mu-5.09*mu**2+4.61*e**2*mu**2


csv = pd.read_csv("kepler_data.csv",delim_whitespace=True)
csv = csv[(csv.Name != "47d")&(csv.Name != "47c")]
 

systems = pd.read_csv("../../Projects/tidal/popsynth/systems.txt", names=["Name","m1","m2","abin","ebin","R1","R2","L1","L2"])

#systems = systems[systems.Name<8]

rows = [row for index, row in systems.iterrows()]

ftide = "f30"

import glob as glob
runs = glob.glob("../../Projects/tidal/popsynth/"+ftide+"/*.p")

#runs = runs[::11]

baseline = 1.5
#baseline = 1.0
#baseline = 1.5

baselines = [0.5,1.0,1.5]

for baseline in baselines:
    plt.figure()
    for run in runs:
        name =  int(run[-7:-2])
        m1 = systems.iloc[name].m1
        m2 = systems.iloc[name].m2
        mu = m2/(m1 + m2)
        #Name = str(int(row['Name'])).zfill(5)
        #print Name
        #df = pd.read_pickle("../../Projects/tidal/popsynth/"+ftide+"/"+Name+".p")
        df = pd.read_pickle(run)
        a = df.iloc[-1].a
        e = df.iloc[-1].e
        hw = hwac(e,mu)
        ap = qs.sma(m1,m2,365*baseline)
        #a0, eo = df.iloc[0].a, df.iloc[0].e
        a0, e0 = systems.iloc[name].abin, systems.iloc[name].ebin
        p0 = qs.period(m1,m2,a0)
        if a*hw > ap:
            #print "Unstable"
            plt.plot(p0,e0,"x",c="red", alpha=0.5)
        else:
            plt.plot(p0,e0,".",c="blue", alpha=0.5)
            
        if df.chaos.min() == -1.0:
            plt.plot(p0,e0,"x",c="orange", alpha=0.5)
        
        elif df.chaos.max() > 0:
            plt.plot(p0,e0,"+",c="green", alpha=0.5)  
            
    kepler = [row1 for index1, row1 in csv.iterrows()]
    for kep in kepler:
        plt.plot(kep.Pbin,kep.ebin,"o",c="orange",)
        plt.text(kep.Pbin,kep.ebin,kep.Name)
    
    plt.ylim(0,.65)
    #plt.xlim([0.025,.25])
    plt.xlabel("$P_{bin}$ (days)")
    plt.ylabel("$e_{bin}$")
    #plt.show()
    plt.title("Max Period = "+str(baseline))
    plt.savefig("/Users/adam/Projects/tidal/popsynth/stable"+str(baseline)+"_N1.png")
    
    #plt.figure()
    #for run in runs:
    #    df = pd.read_pickle(run)
    #    plt.plot(df.a,df.e,alpha=0.5)
    #plt.ylim(0,1)
    #plt.xlim([0.025,.35])
    #plt.xlabel("$a_{bin}$")
    #plt.ylabel("$e_{bin}$")
    #plt.show()
    plt.close()
