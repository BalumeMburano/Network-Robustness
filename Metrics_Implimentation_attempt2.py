# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:52:15 2019

@author: Balume
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy.linalg as np
import numpy as np1
import math 
from scipy.stats import spearmanr

import R_metric
try:
    reload
    
except NameError:
    try:
        from importlib import reload  
    except ImportError:
        from imp import reload 
        

def display_graph(G):
    nx.draw(G)
    plt.show() 
    
    
def natural_connectivity(G):
    n=len(G)
    '''eig_of_aj=nx.laplacian_spectrum(G)
    e_pw_of_eig=[]
    n=len(G)
    for i in eig_of_aj:
        e_pw_of_eig.append(math.pow(math.e,i))
    nat_con=(sum(e_pw_of_eig)/(1/n))'''
    nat_con=(nx.estrada_index(G)*(1/n))
    return nat_con
        
def effectif_resistance(G):
    L=nx.laplacian_matrix(G)
    eg=np.eigvals(L.A)
    n=len(G)
    one_over_eg=[]
    for i in eg[1:]:
        one_over_eg.append(1/i)
    eg_sum=sum(one_over_eg)
    eff_res=eg_sum*n
    return eff_res

def num_of_spanning_tree(G):
    L=nx.laplacian_matrix(G)
    eg=np.eigvals(L.A)
    eg.sort()
    n=len(G)
    er=[]
    for j in eg[1:]:
        er.append(j)
    num_of_span=(np1.prod(er)*(1/n))
    return num_of_span

    
def main():
    try:
       
        graphs=input('Enter the number of graphs : ')
        i=0
        natural=[]
        algebraic=[]
        r_value=[]
        spanning=[]
        eff_resist=[]
        n_graphs=[]
        
        while i in range(int(graphs)):
            ''' reload R_metric'''
            reload(R_metric)
            
            '''Create graph'''
            G=nx.gnm_random_graph(100,400)
            #G=nx.random_regular_graph(5,20,seed=None)
           #G=nx.barabasi_albert_graph(100,50)
            n=len(G)
            #display graphs            
            display_graph(G)
            n_graphs.append(G)
            
            print('Natural connectivity  :',nx.estrada_index(G)*(1/n))
            natural.append(nx.estrada_index(G)*(1/n))
            
            #call  class that return the Algebraic connectivity mmetric
            algebraic.append(nx.algebraic_connectivity(G))
            print('The algebaric Connectivity Value is: ',nx.algebraic_connectivity(G))
            
            
            #call  class that return the Number of Spanning Tree mmetric
            print('Spanning tree is: ',num_of_spanning_tree(G))
            spanning.append(num_of_spanning_tree(G))
            
            print('Effetcive resistance is: ',effectif_resistance(G))
            eff_resist.append(effectif_resistance(G))
            
            #call  class that return the mertic R with node degree as the node importance measure 
           
            R=R_metric.metric_r_degree(G)
            print('The value of R is : ',R)
            r_value.append(R)
            corr_coef, p = spearmanr(r_value,algebraic)
            #The p_value indicates the prpbability of an uncorrealted system of producing datasets that have an extreme spearman correlation  
            print('Spearmans correlation coefficient:', corr_coef)
            print (spearmanr(r_value,algebraic))
            # Sample Interpretation 
            alpha = 0.05
            if p > alpha:
                print('Metrics are uncorrelated  p=%.3f' % p)
            else:
                print('Samples are correlated  p=%.3f' % p)
            i+=1 
            
                
    except Exception as e:
        print(str(e))
        pass
main()

    