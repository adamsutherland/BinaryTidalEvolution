BinaryTidalEvolution

These comments and the code to recreate these plots are in a jupyter notebook but are copied here with the complete figures.

## Tracks for Kepler Systems

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib nbagg
import glob as glob

import imp
qs = imp.load_source('quicksilver', '/home/adam/Code/quicksilver/quicksilver.py')
#reso = imp.load_source('resonances', '/home/adam/Code/resonances/reso.py')
```

`quicksilver` is my code that includes tools for analyzing orbits and calculating resonance widths of planets around binaries. You shouldn't need to edit anything within that code. Just some tools are used to help in this code. For example given a secondary mass of KIC... you can calculate the SMA of the planet and the binary.


```python
a_bin = qs.sma(1.04,0.95,19.39)
print a_bin, qs.mod_a_from_n(1.04,0.95,a_bin,2*np.pi/260)
```

    0.177665845471 1.00473540458


Open `tides.py`

Uncomment:

```python
# multiprocessing for all 
pool = mp.Pool(processes=numcpu)
pool.map(run_dual, rows)
```

This will read a csv file with the Kepler results and ouput the tidal evolution for all systems. Changing `ftide` (line 103) to your required value.

Run tides.py

Plot the results with the code below:


```python
ftide = 50

csv = pd.read_csv("/home/adam/Code/BinaryTidalEvolution/kepler_data.csv",delim_whitespace=True)
csv = csv[(csv.Name != "47d")&(csv.Name != "47c")]
plt.figure(figsize=(10,6))
for index, row in csv.iterrows():
    df = pd.read_pickle("/home/adam/Projects/tidal/f"+str(ftide)+"/"+row['Name']+".p")
    plt.scatter(df.a,df.e,c=df.t,label=row['Name'])
    plt.text(df.a[0]+0.01,df.e[0]-0.02,row['Name'])
    #plt.plot(df.a[df.chaos>2],df.e[df.chaos>2],"r.",label="")
    #plt.clim(cmin,cmax)
#plt.legend(loc="best")
plt.title("$F_{tide}="+str(ftide)+"$")
plt.ylabel("e")
plt.xlabel("a")
plt.xlim(0.05,.45)
plt.colorbar()
plt.show()
```


![Tidal Evolution Tracks](https://raw.githubusercontent.com/adamsutherland/BinaryTidalEvolution/master/plots/tracks.png)


Now open `chaos_check.py` to calculate where the planets are in the N:1. 

Specify `ftide` and `baseline`. Within `check_p_fast` set the planet eccentricity, `ep`, for the resonant widths. Set to the planet's current day eccentricity, the secular eccentricity, or a manual value. Change `df["chaos_ep"]` to some else like `df["chaos_05"]` to track other eccentricites.


```python
plt.figure(figsize=(10,6))
ftide = "f50"

cmin, cmax = 4, 13

for index, row in csv.iterrows():
    df = pd.read_pickle("/home/adam/Projects/tidal/"+ftide+"/"+row['Name']+".p")
    #df.chaos = df.chaos_05
    df.chaos = df.chaos_ep
    plt.plot(df.a,df.e,label=row['Name'])
    plt.scatter(df.a[df.chaos>2],df.e[df.chaos>2],c=df.chaos[df.chaos>2],label="",cmap="rainbow")
    plt.clim(cmin,cmax)
    
plt.title(ftide)
plt.legend(loc="best")
plt.ylabel("e")
plt.xlabel("a")
plt.xlim(0.05,.45)
plt.colorbar()
plt.show()
```


![Tracks with N:1s](https://raw.githubusercontent.com/adamsutherland/BinaryTidalEvolution/master/plots/N1tracks.png)



Now to make the tracks with radius evolution:

Open `tides.py`

comment:

```python
# multiprocessing for all 
pool = mp.Pool(processes=numcpu)
pool.map(run_dual, rows)
```

uncomment:

```python
#limit csv to just values with radius evolution:
csv = csv[(csv.Name == "34b") |  (csv.Name == "38b") | (csv.Name == "47b") | (csv.Name == "KIC")]
rows = [row for index, row in csv.iterrows()]

pool = mp.Pool(processes=numcpu)
pool.map(run_dual_R_evo, rows)
```

run `tides.py`

Optional: rerun `chaos_check.py` after changing the two lines after _limit csv to just values with radius evolution_ and changing check_p_fast to read the `_evo` files. You should be able to see what N:1s the planet is in when plotting the black lines on top of the whole track.


```python
plt.figure(figsize=(10,6))
ftide = "f50"

cmin, cmax = 4, 13


for index, row in csv.iterrows():
    df = pd.read_pickle("/home/adam/Projects/tidal/"+ftide+"/"+row['Name']+".p")
    #df.chaos = df.chaos_05
    df.chaos = df.chaos_ep
    plt.plot(df.a,df.e,label=row['Name'])
    plt.scatter(df.a[df.chaos>2],df.e[df.chaos>2],c=df.chaos[df.chaos>2],label="",cmap="rainbow")
    plt.clim(cmin,cmax)
    
df1 = pd.read_pickle("/home/adam/Projects/tidal/"+ftide+"/"+"34b_evo"+".p")
df2 = pd.read_pickle("/home/adam/Projects/tidal/"+ftide+"/"+"38b_evo"+".p")
df3 = pd.read_pickle("/home/adam/Projects/tidal/"+ftide+"/"+"47b_evo"+".p")
df4 = pd.read_pickle("/home/adam/Projects/tidal/"+ftide+"/"+"KIC_evo"+".p")    

plt.plot(df1.a,df1.e,"k",lw=3,label="")
plt.plot(df2.a,df2.e,"k",lw=3,label="")
plt.plot(df3.a,df3.e,"k",lw=3,label="")
plt.plot(df4.a,df4.e,"k",lw=3,label="")
    
plt.title(ftide)
plt.legend(loc="best")
plt.ylabel("e")
plt.xlabel("a")
plt.xlim(0.05,.45)
plt.colorbar()
plt.show()
```


![Radius Evolution](https://raw.githubusercontent.com/adamsutherland/BinaryTidalEvolution/master/plots/Revo.png)



The time spent in each N:1


```python
import matplotlib.cm as cm

color=cm.rainbow(np.linspace(0,1,5))


plt.figure(figsize=(8,6))
cval = 0
for index, row in csv.iterrows():
    df = pd.read_pickle("/home/adam/Projects/tidal/f50/"+row['Name']+".p")
    for n in xrange(3,11):
        df2 = df[df.chaos_05==n]
        #print row["Name"], n, df2.t.max()-df2.t.min()
        plt.semilogy(n,df2.t.max()-df2.t.min(),".",c=color[cval],ms=10)
    if df.chaos_05.max()>0:
        plt.plot(2,10**7,c=color[cval],label=row['Name'])
        cval +=1
plt.legend(loc="best")
plt.ylabel("Time")
plt.xlabel("N:1")
plt.xlim(2.7,10.3)
plt.show()
```

![Time spent in N:1](https://raw.githubusercontent.com/adamsutherland/BinaryTidalEvolution/master/plots/timeN1.png)



Some planets spend less time in the 4:1 than the 5:1 just because the 5 and 4 overlap and I stop tracking after the stability limit

## Critical Eccentricity Curves

Comment out any lines of than definitions at the end of `tides.py` since when the fuctions get imported to `e_crit.py` the whole file is run. 

Change the eccentricity in `check_p_fast` and the planet period in the same funtion as `baseline`. This program, for a given planet eccentricity and planet period finds the critical binary eccentric for a stable planet. For a particular combination, it runs 12 (the number of threads my computer has) and determines which eccentricties are stable. It then refines the eccentricity. This code takes a while to run. 3 mass ratios X 21 binary periods X (12 eccentricites)^interations. Then run three times for the 3 planet periods.


```python
ecrit = pd.read_csv("/home/adam/Projects/tidal/popsynth/e_crit/meta_results_15.txt",names=["q","P_p","e_crit"])

plt.figure()
plt.scatter(ecrit.P_p,ecrit.e_crit,c=ecrit.q)
plt.colorbar()
plt.show()
```


![Critical Eccentricity](https://raw.githubusercontent.com/adamsutherland/BinaryTidalEvolution/master/plots/results_all_1.5.png)



