"""
@author: Balume
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy.linalg as np
import numpy as np1
from scipy.stats import spearmanr


       

def display_graph(G):
    nx.draw(G)
    plt.show()
    
    
def metric_r_degree(G):
    
        '''Return the metric R with node degree as node importance measure.
	
        This function is specially provided, since calculating node degree is
        quicker	than calculating degree centrality, which is normalzed in NetworkX.

        Parameters
        ----------
        G : Graph
            The graph for which R is to be computed

        Returns
        -------
        R : float
            The robustness metric R of G.
        '''
        
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
    return num_of_span

def spectral_radius(G):
    eig_of_aj=nx.adjacency_spectrum(G)
    spr=max(eig_of_aj)
    return spr

def spectral_gap(G):
    eig_of_aj=nx.adjacency_spectrum(G)
    eig_of_aj.sort()
    sp_g=(eig_of_aj[-1])-(eig_of_aj[-2])
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
        node_con=[]
        #n_graphs=[]
        
        while i in range(int(graphs)):
                        
            '''Create graph'''
            #G=nx.gnm_random_graph(100,400)
            #G=nx.random_regular_graph(5,20,seed=None)
            G=nx.barabasi_albert_graph(100,5)
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
            
            #call  class that return the Algebraic connectivity metric
            algebraic.append(nx.algebraic_connectivity(G))
            print('The algebaric Connectivity Value is: ',nx.algebraic_connectivity(G))
            
            #call  class that return the Number of Spanning Tree metric
            print('Spanning tree is: ',num_of_spanning_tree(G))
            spanning.append(num_of_spanning_tree(G))
            
            #call  class that return the Effective Resistance metric
            
            print('Effetcive resistance is: ',effectif_resistance(G))
            eff_resist.append(effectif_resistance(G))
            
             #call  class that return the Effective Resistance metric
            print('Node connectivity value is: ',nx.node_connectivity(G))
            node_con.append(nx.node_connectivity(G))
            
            #call  class that return the mertic R with node degree as the node importance measure 
            R=metric_r_degree(G)
            
            print('The value of R is : ',R)
            r_value.append(R)
            print('====================================================Graph No: ',i)
            
            i+=1 
        corr_coef_alg, p = spearmanr(r_value,algebraic)
        corr_coef_nat_con, p = spearmanr(r_value,natural)
        corr_coef_spt, p = spearmanr(r_value,spanning)
        corr_coef_ef_res, p = spearmanr(r_value,eff_resist)
        corr_coef_spec_rad, p = spearmanr(r_value,spec_rad)
        corr_coef_gap, p = spearmanr(r_value,spec_gap)
        corr_coef_node, p = spearmanr(r_value,node_con)
        

        
        print('Coefficient for algebraic Is:', corr_coef_alg)
        print('Coefficient for natural connectivity Is:', corr_coef_nat_con)
        print('Coefficient for number of spanning Tree Is:', corr_coef_spt)
        print('Coefficient for effective resistance Is:', corr_coef_ef_res)
        print('Coefficient for spectral radius Is:', corr_coef_spec_rad)
        print('Coefficient for spectral gap Is:', corr_coef_gap)
        print('Coefficient for Node Conn Is:', corr_coef_node)
        
        #Ploting the correlation results
        metrics=['Spectral_Rad','Spectral_gap','Natural_con','Algebraic_con','Spanning_t','Eff_Res','Node_Con']
        pos= np1.arange(len(metrics))
        correl=[corr_coef_spec_rad,corr_coef_gap,corr_coef_nat_con,corr_coef_alg,corr_coef_spt,corr_coef_ef_res,corr_coef_node]
        
        plt.barh(pos,correl,color='blue',edgecolor='black')
        plt.yticks(pos, metrics)
        plt.xlabel('Correlation_Coeficient', fontsize=16)
        plt.ylabel('Metrics', fontsize=16)
        plt.title('Correlation coeficient of Metrics with R_',fontsize=20)
        
        plt.show()
        
       
    except Exception as e:
        print(str(e))
        pass
main()

    