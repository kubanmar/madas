from data_framework import MaterialsDatabase

db = MaterialsDatabase(filename = 'random_database.db')

"""
#read materials ids
mids = []
with open('random_fingerprints.out','r') as f:
    while True:
        line = f.readline()
        if line == '':
            break
        mids.append(line.strip().split(':'))

#add materials to database
for mid in mids:
    db.add_material(mid[0],mid[1])

#add fingerprints
db.add_fingerprint("DOS")
db.add_fingerprint("SOAP")
db.add_fingerprint("IAD")
db.add_fingerprint("PROP")
db.add_fingerprint("SYM")
#generate similarity matrices
for fp_type in ["DOS", "SOAP", "IAD", "PROP", "SYM"]:
    filename = 'random_database_'+fp_type+'_simat.csv'
    matrix = db.get_similarity_matrix(fp_type)
    matrix.save(filename = filename)
"""
