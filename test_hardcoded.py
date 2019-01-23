from data_framework import MaterialsDatabase
import matplotlib.pyplot as plt
import sys

if False:
    test_db = MaterialsDatabase(filename = 'test_db.db')
#test_db = MaterialsDatabase(filename = 'diamond_parent_lattice.db')
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

    matrix.save()

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

if False:
    from similarity import parallel_similarity_search
    import time

    t1 = time.time()

    parallel_similarity_search(test_db.atoms_db, 'DOS', debug = False)

    t2 = time.time()

    print("I took ", t2 - t1, 'time units!')

if False:
    test_db.update_entry('1659:142776',{'test':False})
    row = test_db.atoms_db.get(test = 'false')
    print(row.formula)

if False:
    test_db.add_property_NOMAD('1659:142776','code_version')
    row = test_db.atoms_db.get(1)
    print(row.code_version)

if False:
    print(test_db._get_properties_NOMAD(1659,142776))

if False:
    print(test_db.get_property('1659:142776','atomic_density'))

if False:
    from PROP_Fingerprint import PROPFingerprint
    from PROP_Fingerprint import get_PROP_sym
    GaAs_prop_fp = PROPFingerprint(test_db.atoms_db.get(1).data['properties'])
    next_prop_fp = PROPFingerprint(test_db.atoms_db.get(2).data['properties'])
    print(get_PROP_sym(GaAs_prop_fp, next_prop_fp))

if False:
    test_db.add_fingerprint("PROP")

if False:
    matrix = test_db.get_similarity_matrix('PROP')
    for row in matrix.matrix:
        print(row)

if False:
    from IAD_Fingerprint import IADFingerprint
    GaAs = test_db.atoms_db.get(1).toatoms()
    testfp = IADFingerprint(GaAs)
    print(testfp.get_data())

if False:
    test_db.add_fingerprint("IAD")

if False:
    matrix = test_db.get_similarity_matrix("IAD")
    print(matrix.matrix)

if True:
    from similarity import SimilarityMatrix
    #db = MaterialsDatabase(filename="diamond_parent_lattice.db")
    db = MaterialsDatabase(filename="carbon_oxygen_structures.db")
    #db.add_fingerprint("IAD")
    dos_matrix = SimilarityMatrix()
    dos_matrix.load(filename = 'CO_structures_DOS_simat.csv')
    prop_matrix = SimilarityMatrix()
    prop_matrix.load(filename = 'CO_structures_SOAP_simat.csv')
    prop_matrix.matrix = prop_matrix.get_complement(maximum = 2)
    print('Got all matrices!')
    matrix1, matrix2, mids = dos_matrix.get_matching_matrices(prop_matrix)
    print('Matrices matched.')
    pairs = []
    for idx in range(len(matrix1)):
        for jdx in range(len(matrix1[idx])):
            pairs.append([matrix1[idx][jdx],matrix2[idx][jdx]])
    xs = [x[0] for x in pairs]
    ys = [x[1] for x in pairs]
    import matplotlib.pyplot as plt
    plt.scatter(xs,ys,alpha=0.5, s= 0.5)
    plt.show()

if False:
    GaAs = test_db.atoms_db.get(1)
    from SOAP_fingerprint import SOAPfingerprint
    GaAs_SOAP = SOAPfingerprint(GaAs.toatoms())
    GaAs_SOAP.get_data()

if False:
    #test_db.add_fingerprint('SOAP')
    simat = test_db.get_similarity_matrix("SOAP")
    simat.save("soap_test_db_simat.csv")

if False:
    test_db = MaterialsDatabase('carbon_oxygen_structures.db')
    test_db.add_fingerprint("SOAP")
