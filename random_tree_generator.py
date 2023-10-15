#importing the networkx library
import networkx as nx

#importing the matplotlib library for plotting the graph
import matplotlib.pyplot as plt
import json

data = open("Graph_created_rt.txt",'w') 
data.truncate()

for i in range(100):
    n = 100  # number of nodes
    #seed = 20161  # seed random number generators for reproducibility


    # Use seed for reproducibility
    G = nx.random_tree(n, seed = None, create_using = None)


    # some properties
    print("node degree clustering")
    for v in nx.nodes(G):
        print(f"{v} {nx.degree(G, v)} {nx.clustering(G, v)}")

    print()
    print("the adjacency list")
    for line in nx.generate_adjlist(G):
        print(line)

    pos = nx.spring_layout(G, seed=None)  # Seed for reproducible layout
    nx.draw(G, pos=pos,with_labels=True)
    #plt.show()

    d = nx.to_dict_of_dicts(G)

    # file=open("Graph_created.txt","wb")
    # nx.write_adjlist(d,file,encoding='utf-8')
    # file.close()

    
    data.write(json.dumps(d))
    data.write("\n")

data.close()
