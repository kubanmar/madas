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

if False:
    test_id = test_db.atoms_db.get(1).mid
    print('Got material:', test_id)
    print(test_db.get_property(test_id, 'mainfile_uri'))

if False:
    for row in test_db.atoms_db:
        print(row.formula)

if False:
    test_db.netlog.info('Logging works!')

if False:
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

if False:
    from fingerprints.CLUS_fingerprint import CLUSfingerprint

if False:
    #from similarity import SimilarityMatrix

    import time

    t1 = time.time()

    matrix = test_db.get_similarity_matrix("DOS")#SimilarityMatrix()
    #matrix.calculate("DOS", test_db.atoms_db)

    neighbors_dict = matrix.gen_neighbors_dict()

    t2 = time.time()

    print("I took ", t2 - t1, 'time units!')

    nonsense = input()

    if False:
        import json

        with open('test_matrix_neighbors_dict.json','w') as f:
            json.dump(neighbors_dict,f, indent = 4)

        for key in neighbors_dict.keys():
            print(key)
            for neighbor in neighbors_dict[key]:
                print(neighbor)

        print(len([x for x in neighbors_dict.keys()]))

if False:
    matrix = test_db.get_similarity_matrix("DOS", large = True)#SimilarityMatrix()

if False:
    from similarity import similarity_search

    print(similarity_search(test_db, test_db.atoms_db.get(1).mid, "DOS", k = 2))

if True:
    from similarity import parallel_similarity_search
    import time

    t1 = time.time()

    parallel_similarity_search(test_db.atoms_db, 'DOS', debug = False)

    t2 = time.time()

    print("I took ", t2 - t1, 'time units!')
