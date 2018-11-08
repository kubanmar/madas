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

    def show_cluster(self, cluster_index):
        from ase.visualize import view
        print(self.cpool[cluster_index].get_positions())

""" #From ASE forum, but does not work properly.
def primitive_from_conventional_cell(atoms, spacegroup=1, setting=1):
    #Returns primitive cell given an Atoms object for a conventional
    #cell and it's spacegroup.
    from ase.lattice.spacegroup import Spacegroup
    from ase.utils.geometry  import cut
    sg = Spacegroup(spacegroup, setting)
    prim_cell = sg.scaled_primitive_cell  # Check if we need to transpose
    return cut(atoms, a=prim_cell[0], b=prim_cell[1], c=prim_cell[2])

# Simple test
import ase
from ase.lattice.spacegroup import crystal

# fcc
al = crystal('Al', [(0, 0, 0)], spacegroup=225, cellpar=4.05)
al_prim = primitive_from_conventional_cell(al, 225)

# bcc
fe = crystal('Fe', [(0,0,0)], spacegroup=229, cellpar=2.87)
fe_prim = primitive_from_conventional_cell(fe, 229)
"""
