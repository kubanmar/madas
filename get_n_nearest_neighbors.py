from data_framework import MaterialsDatabase
import sys
import json

#db = MaterialsDatabase(filename = 'test_db.db')
db = MaterialsDatabase(filename = 'diamond_parent_lattice.db')

fp_type = "DOS"

neighbors_dict = {}

neighbors_filename = 'data/DOS_nearest_neighbors_diamond_lattice.json'

sim_matrix, mid_list = db.get_similarity_matrix(fp_type)

n_neighbors = 10

for mid in mid_list:
    neighbors = db.similarity_matrix_row(mid, mid_list, sim_matrix)
    neighbors_zipped = [(sim, idx) for idx, sim in enumerate(neighbors)]
    n_nearest = sorted(neighbors_zipped)[(-1 * n_neighbors + 1):-1] #last n entries without last (because is self-similiarty = 1)
    neighbors_dict[mid] = {x[0]:mid_list[x[1]] for x in n_nearest}

with open(neighbors_filename,'w') as f:
    json.dump(neighbors_dict, f, indent = 4, sort_keys = True)
