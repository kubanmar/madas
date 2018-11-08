from data_framework import MaterialsDatabase

diamond_parent_lattice = False
carbon_oxygen = True

if diamond_parent_lattice:
    db = MaterialsDatabase(filename = 'diamond_parent_lattice.db')

    json_query_diamond = {"search_by":{"exclusive":"0","page":1,"per_page":10},"structure_type":["diamond"],"system_type":["bulk"],"crystal_system":["cubic"],"has_dos":"Yes"}
    json_query_zincblende = {"search_by":{"exclusive":"0","page":1,"per_page":10},"structure_type":["zincblende"],"system_type":["bulk"],"crystal_system":["cubic"],"has_dos":"Yes"}

    db.fill_database(json_query_diamond, tags = ['diamond_parent_lattice', 'diamond'])
    db.fill_database(json_query_zincblende, tags = ['diamond_parent_lattice', 'zincblende'])

if carbon_oxygen:
    db_CO = MaterialsDatabase(filename = 'carbon_oxygen_structures.db')

    json_query = {"search_by":{"element":"C,O","exclusive":"0","page":1,"per_page":10},"has_dos":"Yes","functional_type":["GGA"],"code_name":["VASP"],"system_type":["bulk"]}

    db_CO.fill_database(json_query, tags = ['CO_structures'])
