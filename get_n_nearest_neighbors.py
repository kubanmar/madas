from data_framework import MaterialsDatabase
from utils import plot_similar_dos
import matplotlib.pyplot as plt
import sys
import json

#db = MaterialsDatabase(filename = 'test_db.db')
db = MaterialsDatabase(filename = 'diamond_parent_lattice.db')

fp_type = "DOS"

neighbors_dict = {}

neighbors_filename = 'data/DOS_nearest_neighbors_test.json'

if False:

    sim_matrix, mid_list = db.get_similarity_matrix(fp_type)

    n_neighbors = 10

    for mid in mid_list:
        neighbors = db.similarity_matrix_row(mid, mid_list, sim_matrix)
        neighbors_zipped = [(sim, idx) for idx, sim in enumerate(neighbors)]
        n_nearest = sorted(neighbors_zipped)[(-1 * n_neighbors + 1):-1] #last n entries without last (because is self-similiarty = 1)
        neighbors_dict[mid] = {x[0]:mid_list[x[1]] for x in n_nearest}

    with open(neighbors_filename,'w') as f:
        json.dump(neighbors_dict, f, indent = 4, sort_keys = True)

if True:
    with open(neighbors_filename,'r') as f:
        neighbors_dict = json.load(f)

def neighbors_dict_statistics(neighbors_dict, show = True):
    plt.figure()
    liste = []
    for key in neighbors_dict.keys():
        liste.append(sorted([float(x) for x in neighbors_dict[key].keys()])[-1])
    plt.hist(liste, bins = 100)
    if show:
        plt.show()

neighbors_dict_statistics(neighbors_dict, show = False)

plot_similar_dos('92506:170248', neighbors_dict, db, nmax = 4)
