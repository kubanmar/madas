import matplotlib.pyplot as plt
from bitarray import bitarray
import multiprocessing
import numpy as np
import types, sys
from functools import partial

from fingerprint import Fingerprint
from DOS_fingerprint import DOSFingerprint, Grid
from utils import report_error

def overlap_coeff(matched_fp1,fp2):
    bit_array1, bit_array2 = self.match_fingerprints(fp1, fp2)

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
        self.fingerprints = self.db.gen_fingerprints_list("DOS", log = False, grid_id = self.masks_grid.id)
        similarities = []
        for index, mask in enumerate(self.masks):
            per_mask = []
            with multiprocessing.Pool() as p:
                per_mask = p.map(mask.get_similarity, self.fingerprints)
            similarities.append(per_mask)
            print('Matching DOS band gaps {:.3f} %'.format( (index) / len(self.masks) * 100), end = '\r')
        material_similarities = [[y[x] for y in similarities] for x in range(len(self.fingerprints))]
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

    def update_database(self, mid_gap_list):
        mids = [x[0] for x in mid_gap_list]
        gaps = [{'dos_bandgap': x[1]} for x in mid_gap_list]
        self.db.update_entries(mids, gaps)

    def _gen_masks(self, grid):
        masks = []
        for e_bins in grid.grid():
            masks.append(self._make_dos_mask(dos_cutoff = e_bins[0], grid = grid))
        return masks

    def _make_dos_mask(self, dos_cutoff = 1, grid_id = "dg_cut:2:10:(0, 10)", grid = None):
        dos_mask = [0 if (e[0] < dos_cutoff) else 1 for e in self.masks_grid.grid()]
        mask_string = ''
        for item in dos_mask:
            for n_bin in range(self.masks_grid.num_bins):
                mask_string += '1' if item == 1 else '0'
        byte_mask = bitarray(mask_string).tobytes()
        fp = Fingerprint()
        fp.__class__ = DOSFingerprint
        fp.fp_type = "DOS"
        fp.grid = self.masks_grid
        fp.data = {}
        fp.bins = byte_mask
        fp.indices = [0, len(dos_mask)]
        fp.grid_id = grid_id
        fp.set_similarity_function(self.overlap_similarity)
        return fp

    @staticmethod
    def overlap_similarity(fingerprint1, fingerprint2): #Function to pass to Fingerprint
        #if not hasattr(self, "grid"):
        grid = fingerprint1.grid
        bit_array1, bit_array2 = grid.match_fingerprints(fingerprint1, fingerprint2)
        a = bit_array2.count()
        b = (~bit_array1 & bit_array2).count()
        try:
            s = a / float(a + b)
        except ZeroDivisionError:
            s = 0
            print("NOOOOO!")
            self.log.error('ZeroDivisionError for '+str(self.mid)+' and '+str(fingerprint.mid))
        return s


class SimilarityFunctionScaling():
    """
    Scales a given similarity function using a given scaling function and kwargs.
    **args**:
        * scaling_function: function that maps (0,1) to (0,1)
        * similarity_function: function that returns the similiarty s in (0,1) for two fingerprints
    **kwargs**:
        * passed to scaling_function
    """
    def __init__(self, scaling_function, similarity_function, **kwargs):
        self._scaling_function = partial(scaling_function, **kwargs)
        self._similarity_function = similarity_function

    def similarity(self,fingerprint1, fingerprint2):
        return self._scaling_function(self._similarity_function(fingerprint1, fingerprint2))

def mean_shifted_scaling(x, mean = 0.5):
    return 1- np.tanh((1-x)/mean/2) / np.tanh(1/mean/2)
