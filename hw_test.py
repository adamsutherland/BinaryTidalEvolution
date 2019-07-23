#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:19:03 2019

@author: adam
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import imp
#qs = imp.load_source('quicksilver', '/Users/adam/Code/quicksilver/quicksilver.py')
qs = imp.load_source('quicksilver', '/home/adam/Code/analysis/quicksilver/quicksilver.py')


def hwac(e,mu):
    return 1.60+5.10*e-2.22*e**2+4.12*mu-4.27*e*mu-5.09*mu**2+4.61*e**2*mu**2

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

def check_fast(m1,m2,d,ei,eo,ap):
    f1, f2, f3, f4 = mmm_factors(m1,m2,d)
    n1 = qs.mean_mo(m1,m2,d)
    n = int(n1/mmm_fast(f1,f2,f3,f4,ap))
    if n>40:
        inside = 0
    elif n<2:
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

aps = np.linspace(5.0,1.0,1000)
m1, m2 = 1.0, 0.4
ab, eb = 1.0, 0.2


def stab(m1,m2,ab,eb):
    check1 = 0
    ap1 = 0
    a0 = 0
    for ap in aps:
        es = eb*ab/ap*(m1-m2)/(m1+m2)/0.4115
        ef = 0.04
        ep = ef+es
        check = check_fast(m1,m2,ab,eb,ep,ap)
        #print ap, check
        if check-check1 >1:
            ap1 = a0
            a0 = ap
            #print a0, ap1
        if check1-check == 1.0:
            return a0, ap1, hwac(eb,m2/(m1+m2))
            break
        check1 = check

ebs = np.linspace(0.01,0.9,100)
aos = np.array([])
ans = np.array([])
hws = np.array([])
ebs2 = np.array([])

for eb in ebs:
    result =  stab(m1,m2,ab,eb)
    if result is not None:
        ebs2 = np.append(ebs2,eb)
        aos = np.append(aos,result[0])
        ans = np.append(ans,result[1])
        hws = np.append(hws,result[2])

plt.figure()

plt.plot(aos,ebs2,label="Overlap")
plt.plot(ans,ebs2,label="Island")
plt.plot(hws,ebs2,label="HW")
plt.legend()
plt.title("$m_2/m_1="+str(m2/m1)+"$")
plt.xlabel("$a_p/a_b$")
plt.ylabel("$e_b$")
plt.xlim(2.0,5.0)
plt.savefig("../../Projects/tidal/popsynth/e_crit/hw_"+str(m2/m1)+".png")


plt.show()
