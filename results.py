#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 17:47:42 2019

@author: adam
"""

import numpy as np
import pandas as pd

import imp
#tides = imp.load_source('tides', '/Users/adam/Code/BinaryTidalEvolution/tides.py')
qs = imp.load_source('quicksilver', '/Users/adam/Code/quicksilver/quicksilver.py')

ftide = "f30"
import glob as glob
runs = glob.glob("../../Projects/tidal/popsynth/"+ftide+"/*.p")
runs.sort()


systems = pd.read_csv("../../Projects/tidal/popsynth/systems.txt", names=["Name","m1","m2","abin","ebin","R1","R2","L1","L2"])
systems =  systems[systems.Name>7]


f0 = open("../../Projects/tidal/popsynth/e_crit/popsynthresults.txt","w+")



for name in systems.Name:
    Name = str(int(name)).zfill(5)
    df = pd.read_pickle("../../Projects/tidal/popsynth/"+ftide+"/"+Name+".p")
    name = name - 8
    m1 = systems.iloc[name].m1
    m2 = systems.iloc[name].m2
    a0 = systems.iloc[name].abin
    p0 = qs.period(m1,m2,a0)                     
    f0.write(Name+", "+str(m1)+", "+str(m2)+", "+str(p0)+", ")
    f0.write(str(systems.iloc[name].ebin)+", "+str(df.chaos.max())+", "+str(df.chaos.min())+"\n" )

f0.close()


