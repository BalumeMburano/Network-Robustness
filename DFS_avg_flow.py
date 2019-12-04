# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 09:41:57 2019

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

Graph=nx.Graph(GHT)
pos=nx.spring_layout(Graph)
nx.draw_networkx_nodes(Graph,pos,node_size=700)
nx.draw_networkx_edges(Graph,pos)
nx.draw_networkx_labels(Graph,pos,font_size=20, font_family='sans-serif')

print('========================================')
print(' Gomory and Hu Graph from the Original')

plt.show()


GHT = nx.gomory_hu_tree(G1)
n=len(GHT .nodes)
edges=GHT.edges


all_weights=[]

def dfs_rec(graph, root, visited=None):
    if  visited is None:
        visited=[]
    if root in visited:
        return 
    visited.append(root)
    weights1=[]

    for each in [x for x in graph[root] if x not in visited]:
        b=each
        a=root
        w=graph.get_edge_data(a,b,default=0)
        weights1.append(w)
      
        for values in weights1:
            for keys,val in values.items():
                w_value=int(val)     
        print(a,b,' Weight is:',val)
        all_weights.append(w_value)   
        dfs_rec(graph,each,visited)
    
    return all_weights


i=0
n=len(Graph)
weights=[]
min_weights=[]

for i in range (i,n):
    dd=dfs_rec(Graph,i)
    print('Iteration Number :', i)
    
weights.append(dd)
weights_1=weights[0]
min_weights.append(weights_1[0])

for first,second in zip(weights_1,weights_1[1:]):
    if first > second:
        min_weights.append(second)
    else:
        min_weights.append(first)

print(min_weights,sum(min_weights),len(min_weights),(sum(min_weights)/len(min_weights)))
          
