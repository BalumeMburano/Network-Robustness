
# -*- coding: utf-8 -*-
"""
@author: Balume
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy.linalg as np
import numpy as np1
#import math 
from scipy.stats import spearmanr
#from importlib import reload
#import R_metric
import seaborn as sns
import pandas as pd

       

def display_graph(G):
    nx.draw(G)
    plt.show()
    
    
def metric_r_degree(G):
    
        total = n = G.number_of_nodes()
        for i in range(1, n) :
            node, max_deg = max(G.degree(), key=lambda x: x[1])
            G.remove_node(node)
            c = max(nx.connected_components(G), key=len)
            total += len(c)
        R = total / n / (n+1)
        return R
		
    
    
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
    eg.sort()
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
    #print(eg[0],'and',eg[1])
   # print(eg[0]-eg[1])
    return num_of_span

def spectral_radius(G):
    eig_of_aj=nx.laplacian_spectrum(G)
    eig_of_aj.sort()
    spr=max(eig_of_aj)
    return spr

def spectral_gap(G):
    eig_of_aj=nx.laplacian_spectrum(G)
    eig_of_aj.sort()
    sp_g=min(eig_of_aj)
    return sp_g
    
def main():
    try:
       
        graphs=input('Enter the number of graphs : ')
        i=0
        natural=[]
        algebraic=[]
        r_value=[]
        spanning=[]
        eff_resist=[]
        spec_rad=[]
        spec_gap=[]
        #n_graphs=[]
        
        while i in range(int(graphs)):
                        
            '''Create graph'''
            G=nx.gnm_random_graph(100,400)
            #G=nx.random_regular_graph(5,20,seed=None)
            #G=nx.barabasi_albert_graph(100,5)
            n=len(G)
            #display graphs            
            #display_graph(G)
            #n_graphs.append(G)
            print('The Spectral radius is: ',spectral_radius(G))
            spec_rad.append(spectral_radius(G))
            print('The Spectral Gap is: ',spectral_gap(G))
            spec_gap.append(spectral_gap(G))
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
            R=metric_r_degree(G)
            
            print('The value of R is : ',R)
            r_value.append(R)
            print('====================================================')
            
            i+=1 
        corr_coef_alg, p = spearmanr(r_value,algebraic)
        corr_coef_nat_con, p = spearmanr(r_value,natural)
        corr_coef_spt, p = spearmanr(r_value,spanning)
        corr_coef_ef_res, p = spearmanr(r_value,eff_resist)
        corr_coef_spec_rad, p = spearmanr(r_value,spec_rad)
        corr_coef_gap, p = spearmanr(r_value,spec_gap)
        

        '''
        print('Coefficient for algebraic Is:', corr_coef_alg)
        print('Coefficient for natural connectivity Is:', corr_coef_nat_con)
        print('Coefficient for number of spanning Tree Is:', corr_coef_spt)
        print('Coefficient for effective resistance Is:', corr_coef_ef_res)
        print('Coefficient for spectral radius Is:', corr_coef_spec_rad)
        print('Coefficient for spectral gap Is:', corr_coef_gap)'''
        
        
        #Show the relationship between R-metric and the five metrics connected with line to emphasize continuity
        
        pf_alg={'R_metric':r_value,'Algebraic':algebraic}
        pf_alg_data=pd.DataFrame(data=pf_alg)
        sns.pairplot(pf_alg_data,kind="reg")
        
        
        
        pf_na_con={'R_metric':r_value,'Natural_con':natural}
        pf_nat_con_data=pd.DataFrame(data=pf_na_con)
        sns.pairplot(pf_nat_con_data,kind="reg")
        
        
        
        pf_span_tree={'R_metric':r_value,'spanning_tree':spanning}
        pf_span_tree_data=pd.DataFrame(data=pf_span_tree)
        sns.pairplot(pf_span_tree_data,kind="reg")
        
        
        
        pf_eff_resist={'R_metric':r_value,'Effect_Resist':eff_resist}
        pf_eff_resist_data=pd.DataFrame(data=pf_eff_resist)
        sns.pairplot(pf_eff_resist_data,kind="reg")
        
        
        
        pf_spect_rad={'R_metric':r_value,'Spectral_radius':spec_rad}
        pf_spect_rad_data=pd.DataFrame(data=pf_spect_rad)
        sns.pairplot(pf_spect_rad_data,kind="reg")
        
        
        
        pf_spect_gap={'R_metric':r_value,'Spectral_gap':spec_gap}
        pf_spect_gap_data=pd.DataFrame(data=pf_spect_gap)
        sns.pairplot(pf_spect_gap_data,kind="reg")
        
        
        sns.plt.show()        
    except Exception as e:
        print(str(e))
        pass
main()

    