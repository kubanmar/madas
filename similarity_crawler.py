import json

class SimilarityCrawler():

    def __init__(self, first_members, threshold = 0.9):
        self.members = {}
        for mid in first_members.keys():
            self.members[mid] = first_members[mid]
        self.threshold = threshold

    def reduce_threshold(self, reduce_by = 0.05):
        self.threshold -= reduce_by
        return self.threshold

    def expand(self, neighbors_dict, associates):
        expanded = False
        new_member = None
        used_members = []
        for group in associates:
            used_members = used_members + group
        for mid in self.members.keys():
            for similarity in self.members[mid].keys():
                if float(similarity) >= self.threshold:
                    candidate = self.members[mid][similarity]
                    if not candidate in self.members.keys() and not candidate in used_members:
                        new_member = candidate
                        expanded = True
                        break
        if new_member != None:
            self.members[new_member] = neighbors_dict[new_member]
        return expanded

    def iterate(self, neighbors_dict, associates):
        running = True
        nsteps = 0
        while running:
            running = self.expand(neighbors_dict, associates)
            nsteps += 1
        return nsteps

    def report(self):
        return [x for x in self.members.keys()]

def orphans(neighbors_dict, group_member_list):
    orphans = {}
    full_list = []
    for item in group_member_list:
        for member in item:
            full_list.append(member)
    for mid in neighbors_dict.keys():
        if not mid in full_list:
            orphans[mid] = neighbors_dict[mid]
    return orphans

if __name__ == "__main__":
    neighbors_filename = 'data/DOS_nearest_neighbors_test_earth_mover.json'

    import random
    from data_framework import MaterialsDatabase
    from utils import get_plotting_data
    import matplotlib.pyplot as plt

    db = MaterialsDatabase(filename = "diamond_parent_lattice.db")

    with open(neighbors_filename,'r') as f:
        neighbors_dict = json.load(f)

    groups = []

    associates = []
    iterations = 0
    threshold = 0.95
    while threshold > 0.4:
        changing = True
        while changing:
            changing = False
            iterations += 1
            print("iteration", iterations)
            candidates = orphans(neighbors_dict, associates)
            #mid = random.choice([x for x in candidates.keys()])
            for mid in candidates.keys():
                new_candidate = {}
                new_candidate[mid] = candidates[mid]
                crawler = SimilarityCrawler(new_candidate, threshold = threshold)
                if crawler.iterate(candidates, associates) > 1:
                    associates.append(crawler.report())
                    changing = True
                    break
        print(threshold)
        threshold -= 0.05
        #candidates = orphans(neighbors_dict, associates)

    print(associates)
    print(candidates.keys())
    print(len(associates), len(candidates))
    """
    for orphan in candidates.keys():
        name, energy, dos = get_plotting_data(orphan, db)
        plt.plot(energy, dos, label = name)
        plt.legend()
    plt.show()
    """
    """
    for group in associates:
        print('\nnew group')
        for member in group:
            print(db.get_formula(member))
    """

    for group in associates:
        if len(group) > 2:
            plt.figure()
            for member in group:
                name, energy, dos = get_plotting_data(member, db)
                plt.plot(energy, dos, label = name)
                plt.legend()
            plt.show()

"""
well ... one could run the thing and then check the evolution of the clusters:
    * how does the relationship between threshold and Nmembers evolve?
    * how many outlayers?
"""
