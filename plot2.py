#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:21:52 2019

@author: adam
"""

import pandas as pd
import matplotlib.pyplot as plt

file0 = "../../Projects/tidal/popsynth/e_crit/meta_results_15.txt"
df = pd.read_csv(file0,names=["q","p","ecrit"])
#df2 = pd.read_csv("../../Projects/tidal/popsynth/e_crit/meta_results.txt",names=["q","p","ecrit"])

df = df[df.ecrit<1]
#df = df[df.q==.2]


plt.figure()
plt.scatter(df.p,df.ecrit,c=df.q)
#plt.scatter(df2.p,df2.ecrit,c=df2.q)

plt.colorbar(label="q = [0.2,0.6,1.0]")

plt.title("$P_p = "+file0[-6]+"."+file0[-5]+"$")
plt.ylabel("$e_b$")
plt.xlabel("$P_b$")
plt.ylim([-0.02,0.72])
plt.xlim(0,35)
plt.savefig(file0[:-4]+"_2.png")
plt.show()


