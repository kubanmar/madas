from data_framework import MaterialsDatabase
import sys

test_db = MaterialsDatabase(filename = 'test_db.db')

if False:
    test_db.add_material(1659, 142776) # this -of course- is GaAs
    print(test_db.get_formula(test_db._make_mid(1659, 142776)))
    #test_db.update_database_file()

if False:
    test_json = {"search_by":{"element":"Al,Si,P","exclusive":"0","page":1,"per_page":10},"has_dos":"Yes", "code_name":["VASP"]}
    test_db.fill_database(test_json)

if False:
    test_db.add_fingerprint("DOS")

if False:
    test_db.add_fingerprint("SYM")

if False:
    test_db.add_fingerprint("SOAP")

if False:
    GaAs_id = test_db._make_mid(1659, 142776)
    GaAs_dos_fp = test_db.get_fingerprint(GaAs_id, "DOS")
    GaAs_sym_fp = test_db.get_fingerprint(GaAs_id, "SYM")
    for row in test_db.atoms_db.select():
        print(GaAs_dos_fp.calc_similiarity(row.mid, test_db),GaAs_sym_fp.calc_similiarity(row.mid, test_db))

if False:
    print(test_db.get_random())

if True:
    rnd_str = test_db.get_random(False).toatoms()
    from CLUS_fingerprint import CLUSfingerprint
    new_clus_fp = CLUSfingerprint(rnd_str)
    print(rnd_str)
    print(new_clus_fp._get_pristine_subs())
    new_clus_fp._gen_clusters_pool()
    new_clus_fp.show_cluster(12)
