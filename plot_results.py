#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import csv




molprops = ['HOMO', 'LUMO', 'ET1', 'DH', 'DL']
for prop_name in molprops:
    result_csv = f'results/regression/test_{prop_name}_results.csv'
    with open(result_csv,newline='') as csvfile:
        rdr = csv.reader(csvfile)
        next(rdr) #skip 1st row
        results_data = np.array([list(map(float,row[1:])) for row in rdr]).T

    pred, targets = results_data
    targ_min = np.min(targets)
    targ_max = np.max(targets)


    plt.plot(targets, pred,'ro')
    plt.plot(np.linspace(targ_min,targ_max,100),np.linspace(targ_min,targ_max,100),'k--',lw=0.8) #plot y=x for targets, to guide the eye


    plt.xlabel(prop_name)
    plt.ylabel('Predicted ' + prop_name)
    plt.show()
            
