from clusterx.clusters.clusters_pool import ClustersPool
from clusterx.parent_lattice import ParentLattice
from clusterx.super_cell import SuperCell

from copy import deepcopy
import numpy as np

class CLUSfingerprint():

    def __init__(self, atoms_object, n_points_max = 4):
        if atoms_object != None:
            self.atoms = atoms_object
        self.n_points_max = n_points_max if n_points_max <= len(atoms_object) else len(atoms_object)

    def _gen_clusters_pool(self):
        prist_subs = self._get_pristine_subs()
        plat = ParentLattice(prist_subs[0], prist_subs[1:])
        #plat.serialize()
        scell = SuperCell(plat, np.eye(3))
        npoints = [x for x in range(1,self.n_points_max+1)]
        radii = [-1 for x in range(1,self.n_points_max+1)]
        #radii[0] = 0 # 1 point cluster does not have a radius
        self.cpool = ClustersPool(plat, npoints, radii, scell)

    def _get_pristine_subs(self):
        zs = self.atoms.get_atomic_numbers()
        pristine_atoms = []
        for z in np.unique(zs):
            new_atoms = deepcopy(self.atoms)
            new_atoms.set_atomic_numbers([z for x in zs])
            pristine_atoms.append(new_atoms)
        return pristine_atoms
