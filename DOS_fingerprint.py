import math
import matplotlib.pyplot as ppl
from bitarray import bitarray
import numpy as np
import logging
from fingerprint import Fingerprint

from utils import electron_charge


class DOSFingerprint(Fingerprint):

    def __init__(self, db_row = None, stepsize = 0.05, grid_id = None):
        self.stepsize = stepsize
        self.grid_id = grid_id if grid_id != None else "dg_cut:-2:7:(-10, 5)"
        self._init_from_db_row(db_row)

    def calculate(self, db_row):
        if not hasattr(self, 'grid'):
            self.calculate_grid()
        energy, dos = self._rescale_dos(db_row['data']['dos'], db_row['data']['cell_volume'])
        raw_energy, raw_dos = self._integrate_dos_to_bins(energy, dos)
        self.indices, self.bins = self._calc_byte_fingerprint(raw_energy, raw_dos, self.grid.grid())
        self.grid_id = self.grid.id

    def reconstruct(self, db_row):
        data = self._data_from_db_row(db_row)
        self.bins = bytes.fromhex(data['bins'])
        self.indices = data['indices']
        self.grid_id = data['grid_id']

    def get_data(self):
        data = {}
        data['bins'] = self.bins.hex()
        data['indices'] = self.indices
        data['grid_id'] = self.grid_id
        return data

    def calculate_grid(self):
        self.grid = Grid.create(id=self.grid_id)

    def _rescale_dos(self, dos_object, volume):
        """
        takes as input an 'dos'-object from the NOMAD-api and returns a pair (energy,dos), rescaling the energy to [ev] and the dos to dos/ev/cell
        """
        try:
            dos_object['dos_energies']
        except KeyError:
            dos_object = dos_object['dos']
        energies = []
        dos = []
        #print(volume) #DEBUG
        for idx, energy in enumerate(dos_object['dos_energies']):
            energies.append(energy / electron_charge)
            current_dos = dos_object['dos_values'][0][idx]  # WARNING: the [0] omitts any spin-polarization. Has to be clarified before final implementation.
            if current_dos < 0:
                current_dos = 0
            dos.append(current_dos / 1e-30 / electron_charge) # WARNING! This transformation only (!) applies to VASP calculations
            #print(current_dos, current_dos / volume / electron_charge) #DEBUG
        return energies, dos

    def _interpolate_dos(self, dos_values, energy_values, requested_energy):
        """
        Returns a linearly interpolated value between two DOS points.
        """
        if len(dos_values) != 2 or len(energy_values) != 2:
            raise ValueError("Error in _interpolate_dos: Wrong number of arguments for calculation of gradient.")
        if energy_values[0] == energy_values[1]:
            return dos_values[0]
        gradient = (dos_values[1] - dos_values[0]) / ((energy_values[1] - energy_values[0]) * 1.)
        difference = requested_energy - energy_values[0]
        interpolated = dos_values[0] + gradient * difference
        return interpolated

    def _find_energy_cutoff_indices(self, energy, estart, estop):
        """
        This find the correct indices for the integration, that is, all values of the energy in the interval [estart-1,estop+1] will be in the range [emin_idx,emax_idx]
        WARNING: Assumes that estart<energy[0]!
        """
        emin_idx = None
        emax_idx = None
        index = 0
        while emin_idx is None or emax_idx is None:
            if energy[index] > estart and emin_idx is None:
                emin_idx = index - 1
            if energy[-(index + 1)] < estop and emax_idx is None:
                emax_idx = -index + 1
            index += 1
        if emax_idx == 0:
            emax_idx = None
        return emin_idx, emax_idx

    def _integral_house(self, a, b, interval):
        # integrates a "house" like figure, consisting of a rectangle and a 90 degree triangle, with heights a and b, and width=interval
        return (a + b) * interval / 2.

    def _integrate_dos_to_bins(self, energy, dos):
        """
        Transforms the DOS to a given grid with given step(=bin)size.
        DOS values in these bins are summed and devided by the stepsize.
        If bins are located between two original DOS values, the DOS is linearly interpolated between these values.
        """
        estart = round(int(energy[0] / (self.stepsize * 1.)) * self.stepsize,
                       8)  # define the limits that fit with the predefined stepsize
        estop = round(int(energy[-1] / (self.stepsize * 1.)) * self.stepsize, 8)
        idx_min, idx_max = self._find_energy_cutoff_indices(energy, estart,
                                                            estop)  # cut the energy and dos to fit in the bins, one value is left over for correct interpolation of the dos values at the bin positions
        energy = energy[idx_min:idx_max]
        dos = dos[idx_min:idx_max]
        current_energy = estart
        current_dos = self._interpolate_dos([dos[0], dos[1]], [energy[0], energy[1]], estart)
        dos_binned = []
        energy_binned = []
        index = 1  # starting from the second value, because energy[0]<estart. Thus energy[index] is made to be larger than current_energy.
        while current_energy < estop - self.stepsize:
            energy_binned.append(current_energy)
            next_energy = round(current_energy + self.stepsize, 9)
            integral = 0
            while energy[index] < next_energy:
                integral += self._integral_house(current_dos, dos[index], energy[index] - current_energy)
                current_dos = dos[index]
                current_energy = energy[index]
                index += 1
            next_dos = self._interpolate_dos([dos[index - 1], dos[index]], [energy[index - 1], energy[index]],
                                             next_energy)
            integral += self._integral_house(current_dos, next_dos, next_energy - current_energy)
            dos_binned.append(integral)
            current_energy = next_energy
            current_dos = next_dos
        return energy_binned, dos_binned

    def _binary_bin(self, dos_value, grid_bins):
        bin_dos = ''
        for grid_bin in grid_bins:
            if grid_bin <= dos_value:
                bin_dos += '1'
            else:
                bin_dos += '0'
        return bin_dos

    def _cut_vectors_to_grid_size(self, energy, dos, grid):
        min_e = grid[0][0]
        max_e = grid[-1][0]
        common_min_index = 0
        common_max_index = None
        if energy[0] > min_e and energy[-1] < max_e:
            return energy, dos
        if energy[0] < min_e:
            common_min_index = energy.index(min_e)
        if energy[-1] > max_e:
            common_max_index = energy.index(max_e)
        return energy[common_min_index:common_max_index], dos[common_min_index:common_max_index]

    def _calc_byte_fingerprint(self, energy, dos, grid):
        energy, dos = self._cut_vectors_to_grid_size(energy, dos, grid)
        bin_fp = ''
        grid_index = 0
        for idx, grid_e in enumerate(grid):
            if grid_e[0] > energy[0]:
                grid_index = idx - 1
                if grid_index < 0:
                    grid_index = 0
                break
        grid_start = grid_index
        fp_index = 0
        while grid[grid_index + 1][0] < energy[-1]:
            current_dos = 0
            while energy[fp_index] < grid[grid_index + 1][0]:
                current_dos += dos[fp_index]
                fp_index += 1
            bin_fp += self._binary_bin(current_dos, grid[grid_index][1])
            grid_index += 1
        byte_fp = bitarray(bin_fp).tobytes()
        return [grid_start, grid_index], byte_fp

class Grid():
    """A grid object, specifying the energy/dos grid for the fingerprint generation."""

    @staticmethod
    def create(id=None, mu=-2, sigma=7, grid_type='dg_cut', num_bins=56, cutoff=(-10, 5)):
        self = Grid()
        self.num_bins = num_bins
        if id is None:
            id = Grid().make_grid_id(grid_type, mu, sigma, cutoff)#"%s:%s:%s:%s" % (grid_type, str(mu), str(sigma), str(cutoff))
            self.id = id
            self.grid_type = grid_type
            self.mu = mu
            self.sigma = sigma
            self.cutoff = cutoff
        else:
            self.id = id
            values = id.split(':')
            self.grid_type = values[0]
            self.mu = float(values[1])
            self.sigma = float(values[2])
            self.cutoff = tuple([float(x) for x in values[3][1:-1].split(',')])

        return self

    @staticmethod
    def make_grid_id(grid_type, mu, sigma, cutoff):
        id = "%s:%s:%s:%s" % (grid_type, str(mu), str(sigma), str(cutoff))
        return id

    def grid(self):
        if self.grid_type == 'dg' or self.grid_type == 'double_gauss':
            return self._double_gauss_grid(self.mu, self.sigma, original_stepsize=0.05, y_parameter=0.1)
        elif self.grid_type == 'dg_cut' or self.grid_type == 'double_gauss_cut':
            double_gauss_grid = self._double_gauss_grid(self.mu, self.sigma, original_stepsize=0.05, y_parameter=0.1)
            return self._cut_double_gauss(double_gauss_grid, cutoff=list(self.cutoff))
        else:
            raise ValueError("No valid grid type specified.")

    def _double_gauss_grid(self, mu, sigma, original_stepsize, y_parameter, mu_y=None, sigma_y=None):

        def _gauss(x, mu, sigma, normalized=True):
            coefficient = (math.sqrt(2 * math.pi) * sigma)
            value = math.exp((-0.5) * ((x - mu) / (sigma * 1.)) ** 2)
            if normalized:
                value = value / (coefficient * 1.)
            return value

        def _step_sequencer(x, mu, sigma, original_stepsize):
            return int(round((1 + original_stepsize - _gauss(x, mu, sigma, normalized=False)) / original_stepsize, 9))

        if mu_y is None:
            mu_y = mu
        if sigma_y is None:
            sigma_y = sigma
        asc = 0
        desc = 0
        x_grid = [0]
        while (asc is not None) or (desc is not None):
            if asc is not None:
                asc += _step_sequencer(asc, mu, sigma, original_stepsize) * original_stepsize
                x_grid = x_grid + [round(asc, 8)]
                if asc > 50:
                    asc = None
            if desc is not None:
                desc -= _step_sequencer(desc, mu, sigma, original_stepsize) * original_stepsize
                x_grid = [round(desc, 8)] + x_grid
                if desc < -50:
                    desc = None
        grid = []
        for item in x_grid:
            bins = []
            bin_height = _step_sequencer(item, mu=mu_y, sigma=sigma_y, original_stepsize=y_parameter) / (
                    self.num_bins * 2.)
            for idx in range(1, self.num_bins + 1):
                bins.append(bin_height * idx)
            grid.append([item, bins])
        return grid

    def _cut_double_gauss(self, double_gauss_grid, cutoff):
        cut_grid = []
        for index, item in enumerate(double_gauss_grid):
            if cutoff[0] <= item[0] <= cutoff[1]:
                if double_gauss_grid[index - 1][0] < cutoff[0] < item[0]:
                    cut_grid.append(double_gauss_grid[index - 1])
                cut_grid.append(item)
            elif item[0] > cutoff[1] > double_gauss_grid[index - 1][0]:
                cut_grid.append(item)
            elif item[0] > cutoff[1]:
                break
        return cut_grid

    def plot_grid(self):
        """Input: [[energy1, [bin11,bin12,...]],[energy2,[bin21,bin22,...]]]"""
        import matplotlib.pyplot as ppl
        grid = self.grid()

        for item in grid:
            ppl.plot([item[0], item[0]], [0, item[1][-1]], 'k-')
        for d in range(len(grid[0][1])):  # horrible. expects, that all energies have the same number of bins
            plot_x = []
            plot_y = []
            for item in grid:
                plot_x.append(item[0])
                plot_y.append(item[1][d])
            ppl.plot(plot_x, plot_y, 'k-')
        ppl.show()

    def plot_grid_bars(self, leave_out=False, leave_out_window=(0, 0), show = True, axes = False):
        import matplotlib.pyplot as ppl
        grid = self.grid()
        x = []
        y = []
        width_vec = []
        edgecolor = []
        saved_left_out_height = 0
        for index, item in enumerate(grid):
            if index < len(grid) - 1:
                width = grid[index + 1][0] - item[0]
            else:
                width = abs(item[0] - grid[index - 1][0])
            for height in item[1]:
                if not leave_out or (item[0] < leave_out_window[0] or item[0] > leave_out_window[1]):
                    x.append(item[0])
                    y.append(height)
                    width_vec.append(width)
                    edgecolor.append('black')
                else:
                    if height > saved_left_out_height:
                        saved_left_out_height = height
        if leave_out:
            x.append(leave_out_window[0])
            y.append(saved_left_out_height)
            width_vec.append(abs(leave_out_window[1] - leave_out_window[0]))
            edgecolor.append('grey')

        if axes == None:
            ppl.bar(x, y, width=width_vec, facecolor='none', edgecolor=edgecolor, align='edge', linewidth=1)
        else:
            axes.bar(x, y, width=width_vec, facecolor='none', edgecolor=edgecolor, align='edge', linewidth=1)
        if show:
            ppl.show()

    def match_fingerprints(self, fingerprint1, fingerprint2):
        fp1 = bitarray()
        fp2 = bitarray()
        fp1.frombytes(fingerprint1.bins)
        fp2.frombytes(fingerprint2.bins)
        start_index = max([fingerprint1.indices[0], fingerprint2.indices[0]])
        stop_index = min([fingerprint1.indices[1], fingerprint2.indices[1]])
        # find offsets
        dsp1 = (start_index - fingerprint1.indices[0]) * self.num_bins
        dsp2 = (start_index - fingerprint2.indices[0]) * self.num_bins
        dep1 = (fingerprint1.indices[1] - stop_index) * self.num_bins
        dep2 = (fingerprint2.indices[1] - stop_index) * self.num_bins
        fp1 = fp1[dsp1:len(fp1) - 1 - dep1]
        fp2 = fp2[dsp2:len(fp2) - 1 - dep2]
        return fp1, fp2

    def tanimoto(self, fp1, fp2):
        bit_array1, bit_array2 = self.match_fingerprints(fp1, fp2)
        a = bit_array1.count()
        b = bit_array2.count()
        c = (bit_array1 & bit_array2).count()
        tc = c / float(a + b - c)
        return tc

    def earth_mover_distance(self, fp1, fp2, normalize = True):
        a, b = self.match_fingerprints(fp1, fp2)
        bit_xor = a ^ b
        a_moved = (a & bit_xor).count()
        b_moved = (b & bit_xor).count()
        if a_moved > b_moved:
            distance = a_moved
            norm_length = a.count()
        else:
            distance = b_moved
            norm_length = b.count()
        if normalize:
            distance = distance / norm_length
        return distance

    def earth_mover_similarity(self, a, b):
        return 1 - self.earth_mover_distance(a,b)

    def mutual_information(self,fp1,fp2):
        bit_array1, bit_array2 = self.match_fingerprints(fp1, fp2)
        from sklearn.metrics import normalized_mutual_info_score
        score = normalized_mutual_info_score(bit_array1,bit_array2)
        return score

def get_binary_fingerprint_distribution(db, fp_name = None, bins_offset = 56, normalize = False):
    fp_name = "DOS" if fp_name == None else fp_name
    n_db_entries = db.get_n_entries()
    grid = Grid().create(id = db.get_fingerprint('DOS', fp_name = fp_name, db_id = 1).fingerprint.grid_id)
    n_e_bins = len(grid.grid())
    histogram = np.array([int(x) for x in bitarray(n_e_bins * bins_offset * '0').to01()])
    for db_id in range(1, n_db_entries+1):
        fp = db.get_fingerprint('DOS', fp_name = fp_name, db_id = db_id)
        bins = bitarray()
        bins.frombytes(fp.fingerprint.bins)
        fp_indices = fp.fingerprint.indices
        offset = bitarray(bins_offset * fp_indices[0] * '0')
        end_offset = bitarray(bins_offset * (n_e_bins - fp_indices[1]) * '0')
        histogram_fp = offset + bins + end_offset
        histogram += np.array([int(x) for x in histogram_fp.to01()])
    if normalize:
        histogram = histogram / n_db_entries
    return histogram

def DOS_similarity(fingerprint1, fingerprint2):
    if not hasattr(fingerprint1, 'grid'):
        fingerprint1.calculate_grid()
    return fingerprint1.grid.tanimoto(fingerprint1, fingerprint2)
