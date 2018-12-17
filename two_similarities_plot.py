from data_framework import MaterialsDatabase

db = MaterialsDatabase(filename = 'diamond_parent_lattice.db')
#db = MaterialsDatabase(filename = 'test_db.db')


xs = []
ys = []

mids = [row.mid for row in db.atoms_db.select()]
dos_fps = [db.get_fingerprint(mid, "DOS") for mid in mids]
nentries = len(mids)
#for row1 in db.atoms_db.select():
for idx1 in range(nentries):
    fp = dos_fps[idx1]
    #mid1 =  mids[idx1]
    #fp11 = db.get_fingerprint(mid1, "DOS")
    #fp12 = db.get_fingerprint(mid1, "SYM")
    print(idx1,'of', nentries)
    #for row2 in db.atoms_db.select():
    for idx2 in range(idx1,nentries):
        fp2 = dos_fps[idx2]
        #mid2 = mids[idx2]
        xs.append(fp.get_similarity(fp2))
        ys.append(fp.get_similarity(fp2, s = 'earth_mover'))

import matplotlib.pyplot as plt

plt.scatter(xs,ys)
plt.show()
