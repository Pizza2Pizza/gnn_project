#importing the networkx library
import networkx as nx

#importing the matplotlib library for plotting the graph
import matplotlib.pyplot as plt

from numpy import random
from random import sample

import csv
import json

#open the file to read from
file_from = open("Graph_created.txt",'r')

data_to_write = []

for line in file_from:
    days = random.randint(2,10)
    
    dictionary = json.loads(line)
    G = nx.from_dict_of_dicts(dictionary)

    pos = nx.spring_layout(G)  # Seed for reproducible layout
    nx.draw(G, pos=pos,with_labels=True)
    #plt.show()

    infected = []
    patient_zero = sample(list(G.nodes()),1)
    infected = [patient_zero[0]] #othwerwise we get a reference
    print("Patient zero is :", infected)

    immune = [patient_zero[0]]
    new_infected = []

    for d in range(days):
        for i in infected:
            for n in G.neighbors(i):
                if n not in immune:
                    roll = random.binomial(n=1, p=0.7, size=1)
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
    
    data_to_write.append([dictionary,infected,immune,days,patient_zero])

#writing forwared data into csv file with format of:
#adjacency_list,infected_list,immune_list,n_steps,patient zero
with open('forwarded_graph.csv', 'a', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write multiple rows
    writer.writerows(data_to_write)

#closing file
file_from.close()
    