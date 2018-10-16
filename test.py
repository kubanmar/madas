from data_framework import MaterialsDatabase
import sys

test_db = MaterialsDatabase(filename = 'test_db.json')

if False:
    test_db.add_material(1659, 142776) # this -of course- is GaAs
    test_db.update_database_file()
    test_json = {"search_by":{"element":"Al,Si,P","exclusive":"0","page":1,"per_page":10},"has_dos":"Yes", "code_name":["VASP"]}
    test_db.fill_database(test_json)

if False:
    test_db.add_fingerprint("DOS")

if False:
    GaAs_id = test_db._make_mid(1659, 142776)
    GaAs_dos_fp = test_db.get_fingerprint(GaAs_id, "DOS")
    print(GaAs_dos_fp.calc_similiarity(GaAs_id, test_db))
    for material in test_db.materials_dict.keys():
        print(test_db.get_formula(material),'\t',GaAs_dos_fp.calc_similiarity(material, test_db))

import spglib
from ase.spacegroup import get_spacegroup

GaAs_id = test_db._make_mid(1659, 142776)

for key in test_db.materials_dict.keys():
    print(get_spacegroup(test_db.get_atoms(key), method = 'spglib'))
