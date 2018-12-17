from data_framework import MaterialsDatabase
import matplotlib.pyplot as plt
import sys

#test_db = MaterialsDatabase(filename = 'test_db.db')
test_db = MaterialsDatabase(filename = 'diamond_parent_lattice.db')
print('Loaded database.')

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

if True:
    test_id = test_db.atoms_db.get(1).mid
    print('Got material:', test_id)
    GaAs_dos_fp = test_db.get_fingerprint(test_id, "DOS")
    GaAs_sym_fp = test_db.get_fingerprint(test_id, "SYM")
    GaAs_soap_fp = test_db.get_fingerprint(test_id, "SOAP")
    print('Got fingerprints.')
    for row in test_db.atoms_db.select():
        print(GaAs_dos_fp.calc_similiarity(row.mid, test_db),GaAs_sym_fp.calc_similiarity(row.mid, test_db), GaAs_soap_fp.calc_similiarity(row.mid, test_db))

if False:
    print(test_db.get_random())

if False:
    rnd_str = test_db.get_random(False).toatoms()
    from CLUS_fingerprint import CLUSfingerprint
    new_clus_fp = CLUSfingerprint(rnd_str)
    print(rnd_str)
    print(new_clus_fp._get_pristine_subs())
    new_clus_fp._gen_clusters_pool()
    new_clus_fp.show_cluster(12)

if False:
    simat, mid_list = test_db.get_similarity_matrix("DOS")
    sims = []
    print(test_db.similarity_matrix_row(mid_list[1], mid_list, simat))
    for mid, row in zip(mid_list,simat):
        print(mid, row)
        for item in row:
            sims.append(item)
    sims.sort()
    plt.plot([x for x in range(len(sims))],sims,'.')
#    plt.figure()
#    for item in mid_list:
#        sim_row = test_db.similarity_matrix_row(item, mid_list, simat)
#        plt.plot([x for x in range(len(sim_row))],sim_row, '.-')
    plt.show()
