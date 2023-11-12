#importing the networkx library
import networkx as nx

#importing the matplotlib library for plotting the graph
import matplotlib.pyplot as plt

from numpy import random
from random import sample

import csv
import json
import sys 

type_graph = int(sys.argv[1]) if len(sys.argv) >1 else 1
#1 means ER graph, 2 means rgg graph
# 
folder_name = ""
if type_graph == 1:
    #open the file to read from
    file_from = open("Graph_created_er.txt",'r')
    folder_name = "data/ER_Graph/"
elif type_graph == 2:
    file_from = open("Graph_created_rgg.txt",'r')
    folder_name = "data/RGG_Graph/"
elif type_graph ==3:
    file_from = open("Graph_created_rt.txt",'r')
    folder_name = "data/RandomTree_Graph/"
else:
    raise Exception("Graph is not supported")

data_to_write = []
index = 0

def average_degree(G):
    return sum(G.degree(v) for v in G.nodes) / G.number_of_nodes()

sum_degree = 0
num_Graphs = 0
for line in file_from:
    #days = random.randint(2,10)
    days = 1
    
    dictionary = json.loads(line)
    G = nx.from_dict_of_dicts(dictionary)

    pos = nx.spring_layout(G)  # Seed for reproducible layout
    nx.draw(G, pos=pos,with_labels=True)
    #plt.show()

    infected = []
    patient_zero = sample(list(G.nodes()),1)[0]
    infected = [patient_zero] #othwerwise we get a reference
    print("Patient zero is :", infected)

    immune = [patient_zero]
    new_infected = []

    for d in range(days):
        for i in infected:
            for n in G.neighbors(i):
                if n not in immune:
                    roll = random.binomial(n=1, p=0.3, size=1)
                    if roll == 1:
                        new_infected.append(n)
                        immune.append(n)
                        print("the current node is", n )
            #after checking all neigbours we need delete the "parent node" from infected
            #and move it to the array "seen"

        infected = new_infected
        new_infected = []
    print(immune)
    print(infected)
    index +=1

    #sum over average node degree for all of the graphs
    sum_degree = sum_degree + average_degree(G)
    num_Graphs += 1
    
    data_to_write.append([line,json.dumps(infected),json.dumps(immune),days,patient_zero])

print("Last index:",index)
index_test = round(index*0.8)
#writing forwared data into csv file with format of:
#adjacency_list,infected_list,immune_list,n_steps,patient zero

#average degree for all the graphs 
all_average_degree = sum_degree/num_Graphs

#append all_average_degree to data_to_write
for row in data_to_write:
    row.append(all_average_degree)


with open(folder_name +'/raw/forwarded_graph.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    writer.writerow(["adjacency_list", "infected", "immune","days","patient_zero","av_degree"])
    # write multiple rows
    writer.writerows(data_to_write[0:index_test])


with open(folder_name +'/raw/forwarded_graph_test.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["adjacency_list", "infected", "immune","days","patient_zero","av_degree"])

    # write multiple rows
    writer.writerows(data_to_write[index_test:])


#closing file
file_from.close()

    