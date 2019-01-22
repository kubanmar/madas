import math
import numpy as np
from bitarray import bitarray

class IADFingerprint():

    def __init__(self, atoms, data = None):
        if atoms != None:
            self.cell = atoms.get_cell()
            self.positions = atoms.get_positions()
            self.fingerprint = self.calculate()
        elif data != None:
            self.fingerprint = data
        else:
            raise ValueError('Can not initialize fingerprint without data!')

    def calculate(self, scale_to_unit_cell = True, nbytes_per_bin = 10):
        distances = []
        natoms = len(self.positions)
        positions = self.positions
        cell = self.cell
        if scale_to_unit_cell:
            longest_edge = sum(self.cell)
            scaling = np.linalg.norm(longest_edge)
            positions /= scaling
            cell /= scaling
        for index in range(natoms-1):
            for index2 in range(index+1,natoms):
                distances.append(self._min_distance(positions[index],positions[index2],cell))
        distances = np.array(distances)
        histogram = self._make_hist(distances, normalization = len(distances))
        fingerprint = ''
        for entry in histogram:
            n_bits = nbytes_per_bin * 8
            new_row = np.zeros(n_bits)
            bit_bin_size = 1 / n_bits
            for index in range(n_bits):
                if bit_bin_size * index < entry:
                    new_row[index] = 1
            row = bitarray(new_row.tolist())
            fingerprint +=row.tobytes().hex()
        return fingerprint

    def get_data(self):
        return self.fingerprint

    @staticmethod
    def _make_hist(value_array, xlimits = [0,1], normalization = 1, bins = 100):
        length = abs(xlimits[1] - xlimits[0])
        bin_size = length / bins
        histogram = np.zeros(bins)
        for value in value_array:
            position = int(value / bin_size)
            histogram[position-1] += 1
        histogram /= normalization
        return histogram

    @staticmethod
    def _distance(vec_one,vec_two, round_to = 5):
        if not isinstance(vec_one,(list,np.ndarray)):
            return abs(vec_one-vec_two)
        dist = np.linalg.norm(np.array(vec_one)-np.array(vec_two))
        if round_to != None:
            dist = round(dist,round_to)
        return dist

    @staticmethod
    def _min_distance(v1,v2,crystal): #from CELL
        md = IADFingerprint._distance(v1,v2)
        indices=[]
        indices_minimal=[0,0,0]
        for i in range(-1,2):
            for j in range(-1,2):
                for k in range(-1,2):
                    v2p=[0,0,0]
                    for idx in range(3):
                        v2p[idx] = v2[idx] + i*crystal[0][idx] + j*crystal[1][idx] + k*crystal[2][idx]
                    indices=[i,j,k]
                    d = IADFingerprint._distance(v1,v2p)
                    if d<md:
                        md = d
                        indices_minimal=indices
        return md

def get_IAD_sim(bytes1, bytes2):
    fp1 = bitarray()
    fp2 = bitarray()
    fp1.frombytes(bytes.fromhex(bytes1))
    fp2.frombytes(bytes.fromhex(bytes2))
    a = fp1.count()
    b = fp2.count()
    c = (fp1 & fp2).count()
    tc = c / float(a + b - c)
    return tc
