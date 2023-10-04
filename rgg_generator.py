#importing the networkx library
import networkx as nx

#importing the matplotlib library for plotting the graph
import matplotlib.pyplot as plt
import json

import random
import math

data = open("Graph_created_rgg.txt",'w+') 
data.truncate()

for i in range(100):
    n = 11  # 10 nodes
    seed = 20161  # seed random number generators for reproducibility
    
    while(1):
        radius = random.randint(0,5)
        pos = {i: (random.uniform(0, math.sqrt(n)), random.uniform(0, math.sqrt(n))) for i in range(n)}
        # Use seed for reproducibility
        G = nx.random_geometric_graph(n, radius, pos = pos)
        if not nx.is_connected(G):
            pass
        else:
            break


    # some properties
    print("node degree clustering")
    for v in nx.nodes(G):
        print(f"{v} {nx.degree(G, v)} {nx.clustering(G, v)}")

    print()
    print("the adjacency list")
    for line in nx.generate_adjlist(G):
        print(line)

    pos = nx.spring_layout(G, seed=seed)  # Seed for reproducible layout
    nx.draw(G, pos=pos,with_labels=True)
    #plt.show()

    d = nx.to_dict_of_dicts(G)

    # file=open("Graph_created.txt","wb")
    # nx.write_adjlist(d,file,encoding='utf-8')
    # file.close()

    
    data.write(json.dumps(d))
    data.write("\n")

data.close()