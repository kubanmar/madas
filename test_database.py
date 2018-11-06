from data_framework import MaterialsDatabase

db = MaterialsDatabase(filename = 'diamond_parent_lattice.db')

reference = db.get_random(return_id = True)
print(db.get_formula(reference))
ref_dos_fp = db.get_fingerprint(reference, 'DOS')
ref_sym_fp = db.get_fingerprint(reference, 'SYM')
for d in range(50):
    matid = db.get_random(return_id = True)
    print(db.get_formula(matid),ref_dos_fp.calc_similiarity(matid, db),ref_sym_fp.calc_similiarity(matid, db))
