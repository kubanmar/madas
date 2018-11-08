import multiprocessing
from data_framework import MaterialsDatabase
import numpy as np
import pickle

db = MaterialsDatabase(filename = 'carbon_oxygen_structures.db')
#db = MaterialsDatabase(filename = 'diamond_parent_lattice.db')
#db = MaterialsDatabase(filename = 'test_db.db')

xs = np.array([])
ys = np.array([])

mids = [row.mid for row in db.atoms_db.select()]
nentries = len(mids)
print("collecting DOS fingerprints")
dos_fps = [db.get_fingerprint(mid, "DOS") for mid in mids]
print("collecting SYM fingerprints")
sym_fps = [db.get_fingerprint(mid, "SYM") for mid in mids]
print('finished fp collection')
for index in range(nentries):
    dos_fp = dos_fps[index]
    sym_fp = sym_fps[index]
    with multiprocessing.Pool() as p:
        #print(np.array(p.map(dos_fp.calc_similiarity_multiprocess,[mids[x] for x in range(index+1,nentries)])))
        xs = np.append(xs,np.array(p.map(dos_fp.get_similarity,[dos_fps[x] for x in range(index,nentries)])))
        ys = np.append(ys,np.array(p.map(sym_fp.get_similarity,[sym_fps[x] for x in range(index,nentries)])))
    print('Finished', index, 'out of', nentries)


with open('data/stored_diamond_plat_similarities.dat', 'wb') as f:
    pickle.dump([xs,ys], f)

import matplotlib.pyplot as plt

plt.scatter(xs,ys)
plt.show()
