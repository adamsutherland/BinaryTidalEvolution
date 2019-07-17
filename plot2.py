#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:21:52 2019

@author: adam
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../../Projects/tidal/popsynth/e_crit/meta_results.txt",names=["q","p","ecrit"])

df = df[df.ecrit<1]
#df = df[df.q==.6]


plt.figure()
plt.scatter(df.p,df.ecrit,c=df.q)

plt.colorbar(label="q = [0.2,0.6,1.0]")

plt.title("$P_p = 1.5$")
plt.ylabel("$e_b$")
plt.xlabel("$P_b$")

plt.show()


