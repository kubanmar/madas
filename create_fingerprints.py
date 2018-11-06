from data_framework import MaterialsDatabase

db = MaterialsDatabase(filename = 'diamond_parent_lattice.db')

db.add_fingerprint("DOS")
db.add_fingerprint("SYM")
db.add_fingerprint("SOAP")
