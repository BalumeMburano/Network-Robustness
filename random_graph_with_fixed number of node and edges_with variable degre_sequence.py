# -*- coding: utf-8 -*-
"""
@author: Balume Mburano

"""


import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy.linalg as np
import numpy as np1
from scipy.stats import spearmanr
import statistics as st
import pandas as pd


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
    nat_con=(nx.estrada_index(G)*n)
    return nat_con
        
def effective_resistance(G):
    L=nx.laplacian_matrix(G)
    eg=np.eigvals(L.A)
    eg.sort()
    n=len(G)
    one_over_eg=[]
    for i in eg[1:]:
        one_over_eg.append(1/i)
    eg_sum=sum(one_over_eg)
    eff_res=1/eg_sum*n
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

def variance_of_degrees(G):
    my_tuples=nx.degree(G)
    node_degrees=[x[1] for x in my_tuples]
    variances=st.variance(node_degrees)
    return variances

def seq_generator(n,m,l):
    j=0
    r=0
    y=0
    sequences=[]
    k =round((int(m)/int(n)*2))
    first_range=[k for i in range(int(n))]
    x=round(len(first_range)/2)
    left_half=first_range[:(int(x))]
    right_half=first_range[(int(x)):]
    while y in range(int(l)):
       item=random.choice(left_half)
       item2=random.choice(right_half)
       if item >1:
           item-=1
       if item2<(int(n)):
           item2+=1
       while j in range(len(left_half)):
            left_half[j]=item        
            j+=1
            break 
       while r in range(len(right_half)):
            right_half[r]=item2 
            r+=1
            break
       new_seq=left_half+right_half
       if nx.is_graphical(new_seq):
           sequences.append(new_seq)
           #return sequences
       y+=1
    return sequences
def main():
    number_of_nodes=input("Enter the number of nodes:")
    number_of_edges=input("Enter the number of edges:")
    number_of_sequences=input("Ente the number of degreesequences:")
    #print(seq_generator(number_of_nodes,number_of_edges,number_of_sequences))
    new_sequences=seq_generator(number_of_nodes,number_of_edges,number_of_sequences)
    i=0
    
    natural=[]
    algebraic=[]
    r_value=[]
    spanning=[]
    eff_resist=[]
    spec_rad=[]
    spec_gap=[]
    node_con=[]
    avg_shst_path=[]
    var_of_degrees=[]
    assort_coef=[]
    avg_clustering=[]
    while i in range(int(number_of_sequences)):
        for deg_seq in new_sequences:
            #print(the_sequence)         
            G=nx.random_degree_sequence_graph(deg_seq,tries=10)
            n=len(G)
            #print("The value of z is: ",new_sequences)
            #print(nx.is_graphical(new_sequences[i]))
          
            spec_rad.append(spectral_radius(G))
                
            spec_gap.append(spectral_gap(G))
        
            natural.append(nx.estrada_index(G)*n)
                
            algebraic.append(nx.algebraic_connectivity(G))
              
            spanning.append(num_of_spanning_tree(G))
                
            eff_resist.append(effective_resistance(G))
                
            node_con.append(nx.node_connectivity(G))
       
            avg_shst_path.append(nx.average_shortest_path_length(G,method='dijkstra'))
                
            var_of_degrees.append(variance_of_degrees(G))
                
            assort_coef.append(nx.degree_assortativity_coefficient(G))
                
            avg_clustering.append(nx.average_clustering(G))
                
            R=metric_r_degree(G)
            r_value.append(R)
            #nx.draw(G)
            #break
        i+=1
    
    print('====================================================')
        
    print('printing: ,',i,' graphs correlation values' )
        
    print('====================================================')
    #Spearman Correlation Coefficient of Metrics with R_metrics
        
    r_corr_coef_alg, p = spearmanr(r_value,algebraic)
    r_corr_coef_nat_con, p = spearmanr(r_value,natural)
    r_corr_coef_spt, p = spearmanr(r_value,spanning)
    r_corr_coef_ef_res, p = spearmanr(r_value,eff_resist)
    r_corr_coef_spec_rad, p = spearmanr(r_value,spec_rad)
    r_corr_coef_gap, p = spearmanr(r_value,spec_gap)
    r_corr_coef_node, p = spearmanr(r_value,node_con)
        
        
    #printing the metrics correlation with R        
    df=pd.DataFrame({
            "Metrics":['--------------','Algebraic_con','Natural_con','Number_of_Spa_Trees','Effect_resist','Spectral_Rad','Spectral_Gap','Node_conn'],
            "R_Corr":['-------------',r_corr_coef_alg,r_corr_coef_nat_con,r_corr_coef_spt,r_corr_coef_ef_res,r_corr_coef_spec_rad,r_corr_coef_gap,r_corr_coef_node]
            })
        
    #Ploting the correlation results
    metrics=['Spectral_Rad','Spectral_gap','Natural_con','Algebraic_con','Spanning_t','Eff_Res','Node_Con']
    pos= np1.arange(len(metrics))
    correl=[r_corr_coef_spec_rad,r_corr_coef_gap,r_corr_coef_nat_con,r_corr_coef_alg,r_corr_coef_spt,r_corr_coef_ef_res,r_corr_coef_node]
        
    plt.barh(pos,correl,color='blue',edgecolor='black')
    plt.yticks(pos, metrics)
    plt.xlabel('Correlation_Coefficient', fontsize=16)
    plt.ylabel('Metrics', fontsize=16)
    plt.title('Correlation coefficient of Metrics with R_',fontsize=20)
        
    print(df)
    plt.show()
        
        
    #Spearman Correlation Coefficient of Metrics with Variance
        
    v_corr_coef_alg, p = spearmanr(var_of_degrees,algebraic)
    v_corr_coef_nat_con, p = spearmanr(var_of_degrees,natural)
    v_corr_coef_spt, p = spearmanr(var_of_degrees,spanning)
    v_corr_coef_ef_res, p = spearmanr(var_of_degrees,eff_resist)
    v_corr_coef_spec_rad, p = spearmanr(var_of_degrees,spec_rad)
    v_corr_coef_gap, p = spearmanr(var_of_degrees,spec_gap)
    v_corr_coef_node, p = spearmanr(var_of_degrees,node_con)
    v_corr_coef_r_metric, p = spearmanr(var_of_degrees,r_value)
        
    #printing the metrics correlation with Variance       
        
    df=pd.DataFrame({
            "Metrics":['--------------','Algebraic_con','Natural_con','Number_of_Spa_Trees','Effect_resist','Spectral_Rad','Spectral_Gap','Node_conn'],
            "Variance_Corr":['-------------',v_corr_coef_alg,v_corr_coef_nat_con,v_corr_coef_spt,v_corr_coef_ef_res,v_corr_coef_spec_rad,v_corr_coef_gap,v_corr_coef_node]
            })
        
    #Ploting the correlation results
    metrics=['Spectral_Rad','Spectral_gap','Natural_con','Algebraic_con','Spanning_t','Eff_Res','Node_con','R_metric']
    pos= np1.arange(len(metrics))
    correl=[v_corr_coef_spec_rad,v_corr_coef_gap,v_corr_coef_nat_con,v_corr_coef_alg,v_corr_coef_spt,v_corr_coef_ef_res,v_corr_coef_node,v_corr_coef_r_metric]
        
    plt.barh(pos,correl,color='blue',edgecolor='black')
    plt.yticks(pos, metrics)
    plt.xlabel('Correlation_Coefficient', fontsize=16)
    plt.ylabel('Metrics', fontsize=16)
    plt.title('Correlation coefficient of Metrics with Variance_of_Degrees',fontsize=20)
    print(df)
    plt.show()
        
        
        
    #Spearman Correlation Coefficient of Metrics with Average Shortest Path
        
    asp_corr_coef_alg, p = spearmanr(avg_shst_path,algebraic)
    asp_corr_coef_nat_con, p = spearmanr(avg_shst_path,natural)
    asp_corr_coef_spt, p = spearmanr(avg_shst_path,spanning)
    asp_corr_coef_ef_res, p = spearmanr(avg_shst_path,eff_resist)
    asp_corr_coef_spec_rad, p = spearmanr(avg_shst_path,spec_rad)
    asp_corr_coef_gap, p = spearmanr(avg_shst_path,spec_gap)
    asp_corr_coef_node, p = spearmanr(avg_shst_path,node_con)
    asp_corr_coef_r_metric, p = spearmanr(avg_shst_path,r_value)
        
        
    #printing the metrics correlation with Avaerage Shortest Path          
       
    df=pd.DataFrame({
            "Metrics":['----------','Algebraic_con','Natural_con','Number_of_Spa_Trees','Effect_resist','Spectral_Rad','Spectral_Gap','Node_conn'],
            "Average_Sh_Path_Corr":['--------------------',asp_corr_coef_alg,asp_corr_coef_nat_con,asp_corr_coef_spt,asp_corr_coef_ef_res,asp_corr_coef_spec_rad,asp_corr_coef_gap,asp_corr_coef_node]
            })
        
        
        
    #Ploting the correlation results
    metrics=['Spectral_Rad','Spectral_gap','Natural_con','Algebraic_con','Spanning_t','Eff_Res','Node_con','R_metric']
    pos= np1.arange(len(metrics))
    correl=[asp_corr_coef_spec_rad,asp_corr_coef_gap,asp_corr_coef_nat_con,asp_corr_coef_alg,asp_corr_coef_spt,asp_corr_coef_ef_res,asp_corr_coef_node,asp_corr_coef_r_metric]
        
    plt.barh(pos,correl,color='blue',edgecolor='black')
    plt.yticks(pos, metrics)
    plt.xlabel('Correlation_Coefficient', fontsize=16)
    plt.ylabel('Metrics', fontsize=16)
    plt.title('Correlation coefficient of Metrics with Average_Shortest_Path',fontsize=20)
    print(df)
    plt.show()
        
        
    #Spearman Correlation Coefficient of Metrics with Assortative Coeffiecient
        
    assort_corr_coef_alg, p = spearmanr(assort_coef,algebraic)
    assort_corr_coef_nat_con, p = spearmanr(assort_coef,natural)
    assort_corr_coef_spt, p = spearmanr(assort_coef,spanning)
    assort_corr_coef_ef_res, p = spearmanr(assort_coef,eff_resist)
    assort_corr_coef_spec_rad, p = spearmanr(assort_coef,spec_rad)
    assort_corr_coef_gap, p = spearmanr(assort_coef,spec_gap)
    assort_corr_coef_node, p = spearmanr(assort_coef,node_con)
    assort_corr_coef_r_metric, p = spearmanr(assort_coef,r_value)
        
        
    #printing the metrics correlation with Assortative Coeff Cor         
        
        
    df=pd.DataFrame({
            "Metrics":['----------','Algebraic_con','Natural_con','Number_of_Spa_Trees','Effect_resist','Spectral_Rad','Spectral_Gap','Node_conn'],
            "Assortstive_Corr":['---------------',assort_corr_coef_alg,assort_corr_coef_nat_con,assort_corr_coef_spt,assort_corr_coef_ef_res,assort_corr_coef_spec_rad,assort_corr_coef_gap,assort_corr_coef_node]
            })
        
    #Ploting the correlation results
    metrics=['Spectral_Rad','Spectral_gap','Natural_con','Algebraic_con','Spanning_t','Eff_Res','Node_con','R_metric']
    pos= np1.arange(len(metrics))
    correl=[assort_corr_coef_spec_rad,assort_corr_coef_gap,assort_corr_coef_nat_con,assort_corr_coef_alg,assort_corr_coef_spt,assort_corr_coef_ef_res,assort_corr_coef_node,assort_corr_coef_r_metric]
        
    plt.barh(pos,correl,color='blue',edgecolor='black')
    plt.yticks(pos, metrics)
    plt.xlabel('Correlation_Coefficient', fontsize=16)
    plt.ylabel('Metrics', fontsize=16)
    plt.title('Correlation coefficient of Metrics with Assortative Coefficient',fontsize=20)
    print(df)
    plt.show()
        
    #Spearman Correlation Coefficient of Metrics with Clustering Coefficient
        
    clust_corr_coef_alg, p = spearmanr(avg_clustering,algebraic)
    clust_corr_coef_nat_con, p = spearmanr(avg_clustering,natural)
    clust_corr_coef_spt, p = spearmanr(avg_clustering,spanning)
    clust_corr_coef_ef_res, p = spearmanr(avg_clustering,eff_resist)
    clust_corr_coef_spec_rad, p = spearmanr(avg_clustering,spec_rad)
    clust_corr_coef_gap, p = spearmanr(avg_clustering,spec_gap)
    clust_corr_coef_node, p = spearmanr(avg_clustering,node_con)
    clust_corr_coef_r_metric, p = spearmanr(avg_clustering,r_value)
        
        
    #printing the metrics correlation with Clustering Coeff
        
    df=pd.DataFrame({
            "Metrics":['----------','Algebraic_con','Natural_con','Number_of_Spa_Trees','Effect_resist','Spectral_Rad','Spectral_Gap','Node_conn'],
            "Clustering_Corr":['---------------',clust_corr_coef_alg,clust_corr_coef_nat_con,clust_corr_coef_spt,clust_corr_coef_ef_res,clust_corr_coef_spec_rad,clust_corr_coef_gap,clust_corr_coef_node]
            })
        
    #Ploting the correlation results
    metrics=['Spectral_Rad','Spectral_gap','Natural_con','Algebraic_con','Spanning_t','Eff_Res','Node_con','R_metric']
    pos= np1.arange(len(metrics))
    correl=[clust_corr_coef_spec_rad,clust_corr_coef_gap,clust_corr_coef_nat_con,clust_corr_coef_alg,clust_corr_coef_spt,clust_corr_coef_ef_res,clust_corr_coef_node,clust_corr_coef_r_metric]
        
    plt.barh(pos,correl,color='blue',edgecolor='black')
    plt.yticks(pos, metrics)
    plt.xlabel('Correlation_Coefficient', fontsize=16)
    plt.ylabel('Metrics', fontsize=16)
    plt.title('Correlation coefficient of Metrics with Clustering Coefficient',fontsize=20)
    print(df)
    plt.show()
        
    df=pd.DataFrame({
            "Metrics":['----------','Algebraic_con','Natural_con','Number_of_Spa_Trees','Effect_resist','Spectral_Rad','Spectral_Gap','Node_conn'],
            "Ast_Corr":['---------------',assort_corr_coef_alg,assort_corr_coef_nat_con,assort_corr_coef_spt,assort_corr_coef_ef_res,assort_corr_coef_spec_rad,assort_corr_coef_gap,assort_corr_coef_node],
            "Av_Corr":['--------------------',asp_corr_coef_alg,asp_corr_coef_nat_con,asp_corr_coef_spt,asp_corr_coef_ef_res,asp_corr_coef_spec_rad,asp_corr_coef_gap,asp_corr_coef_node],
            "Cl_Corr":['---------------',clust_corr_coef_alg,clust_corr_coef_nat_con,clust_corr_coef_spt,clust_corr_coef_ef_res,clust_corr_coef_spec_rad,clust_corr_coef_gap,clust_corr_coef_node],
            "Var_Corr":['-------------',v_corr_coef_alg,v_corr_coef_nat_con,v_corr_coef_spt,v_corr_coef_ef_res,v_corr_coef_spec_rad,v_corr_coef_gap,v_corr_coef_node],
            "R_Corr":['-------------',r_corr_coef_alg,r_corr_coef_nat_con,r_corr_coef_spt,r_corr_coef_ef_res,r_corr_coef_spec_rad,r_corr_coef_gap,r_corr_coef_node]
            })
    df.to_csv('DegreeSequenceResults_r.csv')
    
main()