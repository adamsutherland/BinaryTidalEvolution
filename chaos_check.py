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

ftide = "f30"


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
        ap = 1.7
        ep = row2["e"]*row2["a"]/ap*(row["m1"]-row["m2"])/(row["m1"]+row["m2"])/0.4115
        #ep = row["ep"]
        chaos = np.append(chaos,check(row["m1"],row["m2"],row2["a"],row2["e"],ep,ap))
    df["chaos"] = chaos
    print chaos.max()
    df.to_pickle("../../Projects/tidal/popsynth/"+ftide+"/"+Name+".p")

import multiprocessing as mp

systems = qs.pd.read_csv("../../Projects/tidal/popsynth/systems.txt", names=["Name","m1","m2","abin","ebin","R1","R2","L1","L2"])

systems = systems[systems.Name<3]
systems = systems[systems.Name>1]


numcpu = mp.cpu_count()
rows = [row for index, row in systems.iterrows()]
#
#if len(rows) % numcpu != 0:
#    batches = len(rows)/numcpu+1
#    numcpu = len(rows)/batches+1
#
#pool = mp.Pool(processes=numcpu)
#pool.map(check_p, rows)

for row in rows:
    Name = str(int(row['Name'])).zfill(5)
    print Name
    df = pd.read_pickle("../../Projects/tidal/popsynth/"+ftide+"/"+Name+".p")
    chaos = np.array([])
    for index2, row2 in df.iterrows():
        ap = 1.7
        ep = row2["e"]*row2["a"]/ap*(row["m1"]-row["m2"])/(row["m1"]+row["m2"])/0.4115
        print row2["a"]
        #ep = row["ep"]
        chaos = np.append(chaos,check(row["m1"],row["m2"],row2["a"],row2["e"],ep,ap))
    df["chaos"] = chaos
    print chaos.max()
    df.to_pickle("../../Projects/tidal/popsynth/"+ftide+"/"+Name+".p")







