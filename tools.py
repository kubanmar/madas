import matplotlib.pyplot as plt
from bitarray import bitarray
from utils import plot_FP_in_grid
import multiprocessing
import numpy as np
import types, sys

from DOS_fingerprints import Grid
from fingerprints import Fingerprint
from utils import report_error

def overlap_coeff(self,fp1,fp2): #Funtion to pass to Grid
    bit_array1, bit_array2 = self.match_fingerprints(fp1, fp2)
    a = bit_array2.count()
    b = (~bit_array1 & bit_array2).count()
    s = a / float(a + b)
    return s

class BandGapFinder():

    def __init__(self, database, logger = None):
        self.db = database
        self.fingerprints = []
        self.masks_grid = Grid.create(mu = 2, sigma = 10, cutoff = (0,10))
        self.masks = self._gen_masks(self.masks_grid)
        if hasattr(database, 'log'):
            self.log = database.log
        else:
            self.log = logger

    def get_dos_gaps(self):
        self.fingerprints = self._get_fingerprints()
        similarities = []
        for mask in self.masks:
            per_mask = []
            def overlap_sim(fingerprint):
                pass
            #with multiprocessing.Pool() as p:
            #    per_mask = p.map(mask.get_similarity, self.fingerprints)
            for fingerprint in self.fingerprints:
                per_mask.append(mask.get_similarity(fingerprint))
            similarities.append(per_mask)
        material_similarities = [[y[x] for y in similarities] for x in range(len(fingerprints))]
        groups = []
        for idx, sim in enumerate(material_similarities):
            found = False
            largest_index = 0
            for jdx, sim_val in enumerate(sim):
                if sim_val == 1.0:
                    largest_index = jdx
                else:
                    found = True
                    break
            groups.append(largest_index)
        e_values = [x[0] for x in self.masks_grid.grid()]
        dos_bandgaps = [e_values[x] for x in groups]
        mids = [fingerprint.mid for fingerprint in self.fingerprints]
        return [[mid, gap] for mid,gap in zip(mids, dos_bandgaps)]

    def _gen_masks(self, grid):
        masks = []
        for e_bins in grid.grid():
            masks.append(self._make_dos_mask(dos_cutoff = e_bins[0], grid = grid))
        return masks

    def _make_dos_mask(self, dos_cutoff = 1, grid_id = "dg_cut:2:10:(0, 10)", grid = None):
        grid = Grid().create(id = grid_id) if grid == None else grid
        dos_mask = [0 if (e[0] < dos_cutoff) else 1 for e in grid.grid()]
        mask_string = ''
        for item in dos_mask:
            for n_bin in range(grid.num_bins):
                mask_string += '1' if item == 1 else '0'
        byte_mask = bitarray(mask_string).tobytes()
        fp = Fingerprint("DOS", calculate = False, log = False)
        fp.data = {}
        fp.data['bins'] = byte_mask.hex()
        fp.data['indices'] = [0, len(dos_mask)]
        fp.data['grid_id'] = grid_id
        fp._reconstruct_from_data()
        fp.set_similarity_function(self.overlap_similarity)
        return fp

    @staticmethod
    def overlap_similarity(self, fingerprint): #Function to pass to Fingerprint
        #if not hasattr(self, "grid"):
        self.grid = Grid().create(id = "dg_cut:2:10:(0, 10)")
        if not hasattr(self.grid,"overlap_coeff"):
            #setattr(self, overlap_coeff, function)
            self.grid.overlap_coeff = types.MethodType(overlap_coeff, self.grid)
        try:
            similarity = self.grid.overlap_coeff(self.fingerprint, fingerprint.fingerprint)
        except ZeroDivisionError:
            similarity = 0
            print("NOOOOO!")
            if self.log != None:
                self.log.error('ZeroDivisionError for '+str(self.mid)+' and '+str(fingerprint.mid))
        return similarity

    def _get_fingerprints(self):
        fingerprints = []
        with self.db.atoms_db as db:
            for row_id in range(1,db.count()+1):
                fingerprint = None
                row = db.get(row_id)
                try:
                    fingerprint = Fingerprint("DOS", mid = row.mid, properties = row.data.properties, atoms = row.toatoms(), mu = 2, sigma = 10, cutoff = (0,10), fp_name = 'band_gap_finder', log = False)
                except AttributeError:
                    fingerprint = Fingerprint("DOS", mid = row.mid, properties = row.data, atoms = row.toatoms(), mu = 2, sigma = 10, cutoff = (0,10), fp_name = 'band_gap_finder', log = False)
                except:
                    report_error(self.log, 'could not get fingerprint for material ' + str(row.mid))
                    continue
                if fingerprint != None:
                    fingerprints.append([row.id, fingerprint])
        return fingerprints
