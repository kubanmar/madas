from data_framework import MaterialsDatabase

db = MaterialsDatabase(filename = 'diamond_parent_lattice.db')

json_query_diamond = {"search_by":{"exclusive":"0","page":1,"per_page":10},"structure_type":["diamond"],"system_type":["bulk"],"crystal_system":["cubic"],"has_dos":"Yes"}
json_query_zincblende = {"search_by":{"exclusive":"0","page":1,"per_page":10},"structure_type":["zincblende"],"system_type":["bulk"],"crystal_system":["cubic"],"has_dos":"Yes"}

db.fill_database(json_query_diamond, tags = ['diamond_parent_lattice', 'diamond'])
db.fill_database(json_query_zincblende, tags = ['diamond_parent_lattice', 'zincblende'])
