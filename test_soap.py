from data_framework import MaterialsDatabase
from SOAP_fingerprint import SOAPfingerprint

db = MaterialsDatabase(filename = 'test_soap.db')

db.add_material(nomad_material_id = 51, nomad_calculation_id = 254634) #diamond Si
db.add_material(nomad_material_id = 38, nomad_calculation_id = 65468) #diamond Ge
db.add_material(nomad_material_id = 210099, nomad_calculation_id = 383280) # zincblende SiGe

soaps = []
for row in db.select():
        pass
