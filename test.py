from data_framework import MaterialsDatabase

test_db = MaterialsDatabase(filename = 'test_db.json')

test_db.add_material(129304, 249843) # this -of course- is GaAs

#print(test_db.materials_dict)

test_db.update_database_file()
