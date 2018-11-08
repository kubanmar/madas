from data_framework import MaterialsDatabase

#db = MaterialsDatabase(filename = 'diamond_parent_lattice.db')
db = MaterialsDatabase(filename = 'test_db.db')


xs = []
ys = []

mids = [row.mid for row in db.atoms_db.select()]
nentries = len(mids)
#for row1 in db.atoms_db.select():
for idx1 in range(nentries):
    mid1 =  mids[idx1]
    fp11 = db.get_fingerprint(mid1, "DOS")
    fp12 = db.get_fingerprint(mid1, "SYM")
    print(mids.index(mid1) + 1,'of', nentries)
    #for row2 in db.atoms_db.select():
    for idx2 in range(idx1,nentries):
        mid2 = mids[idx2]
        if mid1 != mid2:
            xs.append(fp11.calc_similiarity(mid2, db))
            ys.append(fp12.calc_similiarity(mid2, db))

import matplotlib.pyplot as plt

plt.scatter(xs,ys)
plt.show()
