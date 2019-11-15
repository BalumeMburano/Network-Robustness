# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:53:11 2019

@author: 90937669
"""

import networkx as nx
import matplotlib.pyplot as plt


G1=nx.Graph()

G1.add_edge(0, 1, capacity=10.0)
G1.add_edge(1, 0, capacity=13.0)
G1.add_edge(0,2, capacity=9.0)
G1.add_edge(2,0, capacity=6.0)
G1.add_edge(0,3, capacity=11.0)
G1.add_edge(3,0, capacity=5.0)
G1.add_edge(0,4, capacity=7.0)
G1.add_edge(1,3, capacity=8.0)
G1.add_edge(2,3, capacity=3.0)
G1.add_edge(1,4, capacity=15.0)
G1.add_edge(4,3, capacity=2.0)
G1.add_edge(4,5, capacity=3.0)
G1.add_edge(4,6, capacity=11.0)

GHT = nx.gomory_hu_tree(G1)
mins=[]
n=len(GHT .nodes)
edges=GHT.edges
weights=[]
weights_values=[]

for a,b in edges:
    weights.append(GHT.get_edge_data(a,b,default=0))

for values in weights:
    for keys,val in values.items():
        weights_values.append(int(val))
print('The Graph edges are: ',edges)
print('The weight of the edges are: ',weights_values) 
path = nx.shortest_path(GHT, 0, 6, weight='weight') 
print(path)     

G=nx.Graph(GHT)
pos=nx.spring_layout(G)
nx.draw_networkx_nodes(G,pos,node_size=700)
nx.draw_networkx_edges(G,pos)
nx.draw_networkx_labels(G,pos,font_size=20, font_family='sans-serif')

print('========================================')
print(' Gomory and Hu Graph from the Original')

plt.show()

def minimum_edge_weight_in_shortest_path1(GHT, u, v):
    path = nx.shortest_path(GHT, u, v, weight='weight')
    return min((GHT[u][v]['weight'], (u,v)) for (u, v) in zip(path, path[1:]))
i=0
j=0
l=len(GHT.nodes)
degrees=[]

for i in range (i,l-1):
    j=i+1
    u=i
    for j in range (j,l):
        
        v=j
        cut_value, edge = minimum_edge_weight_in_shortest_path1(GHT, u, v)
        degrees.append(cut_value)
        print(i,j)
        print('Path',i,' is :',path)
        print('For Path',i,' : the cut value is:',cut_value,'and the min cut value',nx.minimum_cut_value(G1, u, v))

    
tot=sum(degrees)
print(path)
n=len(GHT)
comb=((n-1)*n)/2
print(tot,comb)
print('The average is:',tot/comb)
