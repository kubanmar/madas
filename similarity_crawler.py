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

    def expand(self, neighbors_dict):
        expanded = False
        new_member = None
        for mid in self.members.keys():
            for similarity in self.members[mid].keys():
                if float(similarity) >= self.threshold:
                    candidate = self.members[mid][similarity]
                    if not candidate in self.members.keys():
                        new_member = candidate
                        expanded = True
                        break
        if new_member != None:
            self.members[new_member] = neighbors_dict[new_member]
        return expanded

    def iterate(self, neighbors_dict):
        running = True
        nsteps = 0
        while running:
            running = self.expand(neighbors_dict)
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
    neighbors_filename = 'data/DOS_nearest_neighbors_test.json'

    import random
    from data_framework import MaterialsDatabase

    with open(neighbors_filename,'r') as f:
        neighbors_dict = json.load(f)

    groups = []

    associates = []

    iterations = 0
    while iterations < 10000:
        iterations += 1
        candidates = orphans(neighbors_dict, associates)
        new_candidate = {}
        mid = random.choice([x for x in candidates.keys()])
        new_candidate[mid] = candidates[mid]
        crawler = SimilarityCrawler(new_candidate, threshold = 0.9)
        if crawler.iterate(candidates) > 1:
            associates.append(crawler.report())
        candidates = orphans(neighbors_dict, associates)

    print(associates)

    db = MaterialsDatabase(filename = "diamond_parent_lattice.db")
    """
    for group in associates:
        print('\nnew group')
        for member in group:
            print(db.get_formula(member))
    """
    from utils import get_plotting_data
    import matplotlib.pyplot as plt

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
