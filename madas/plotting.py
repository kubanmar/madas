import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from bitarray import bitarray

from ase.visualize.plot import plot_atoms
from ase.build import make_supercell

from madas.material import Material 

def plot_minmax(target):
    plt.plot([min(target), max(target)],[min(target), max(target)],'--')

def parity_plot(pred, target, symbol = 'o', figsize = (7,7), show = True):
    plt.figure(figsize = figsize)
    plt.title('RMSE: ' + str(round(np.sqrt(mean_squared_error(pred, target)), 3)), fontsize = 35)
    max_x = max([max(pred), max(target)])*1.1
    min_x = min([min(pred), min(target)])*1.1
    plt.scatter(pred, target, marker=symbol, edgecolor = 'blue')
    plt.plot([min_x, max_x], [min_x, max_x], '--', linewidth=3, color = 'orange')
    plt.xlim(min_x, max_x)
    plt.ylim(min_x, max_x)
    plt.ylabel('Target', fontsize = 35)
    plt.xlabel('Prediction', fontsize = 35)
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    axes = plt.gca()
    plt.setp(axes.spines.values(), linewidth=3)
    axes.xaxis.set_tick_params(width=3, length = 10)
    axes.yaxis.set_tick_params(width=3, length = 10)
    if show:
        plt.show()

def plot_FP_in_grid(byte_fingerprint, grid, show_grid = True, show = True, label = '', axes = None, **kwargs):
    # This should go to nomad-dos-fingerprints
    x=[]
    y=[]
    all_width=[]
    bin_fp=bitarray()
    bin_fp.frombytes(byte_fingerprint.bins)
    grid_indices=byte_fingerprint.indices
    plotgrid=grid.grid()
    plotgrid=plotgrid[grid_indices[0]:grid_indices[1]]
    bit_position=0
    for index,item in enumerate(plotgrid):
        if index<len(plotgrid)-1:
            width=plotgrid[index+1][0]-item[0]
        else:
            width=abs(item[0]-plotgrid[index-1][0])
        for dos_value in item[1]:
            if bin_fp[bit_position]==1 and bin_fp[bit_position+1]==0:
                x.append(item[0])
                y.append(dos_value)
                all_width.append(width)
            bit_position+=1
    if axes is None:
        plt.bar(x,y,width=all_width,align='edge', label = label, **kwargs)
    else:
        axes.bar(x,y,width=all_width,align='edge', label = label, **kwargs)
    if show_grid:
        grid.plot_grid_bars(show = False, axes = axes, **kwargs)
    if show:
        plt.show()

def sub_numbers(string, bold = False):
    subbed_string = ''
    last_was_number = False
    for char in string:
        try:
            float(char)
            if not last_was_number:
                subbed_string += '_{'
            subbed_string += char
            last_was_number = True
        except ValueError:
            if last_was_number:
                subbed_string += '}'
            subbed_string += char
            last_was_number = False
    if last_was_number:
        subbed_string += '}'
    if bold:
        return  r'$\mathbf{'+subbed_string+r'}$'
    return  r'$\mathrm{'+subbed_string+r'}$'

def plot_material(material: Material, 
                  repeat = [2,2,2], 
                  rotation = '-66x,-2y,-6z', 
                  radii = None, 
                  show = True, 
                  show_unit_cell = 0) -> None:
    supercell = make_supercell(material.atoms, [[repeat[0],0,0], [0,repeat[1],0], [0,0,repeat[2]]])
    plot_atoms(supercell, rotation = rotation, radii = radii, show_unit_cell = show_unit_cell)
    ax = plt.gca()
    plt.setp(ax.spines.values(), visible=False) # remove outer spines
    ax.tick_params(left=False, labelleft=False)
    ax.tick_params(bottom=False, labelbottom=False)
    if show:
        plt.show()

def plot_structure_by_mid(mid, database, repeat = [3,3,1], rotation = '-66x,-2y,-6z', radii = None, show = True, return_atoms = False, show_unit_cell = 0):
    atoms = database[mid].atoms
    supercell = make_supercell(atoms, [[repeat[0],0,0], [0,repeat[1],0], [0,0,repeat[2]]])
    plot_atoms(supercell, rotation = rotation, radii = radii, show_unit_cell = show_unit_cell)
    plt.title(sub_numbers(atoms.get_chemical_formula()), fontsize = 20)
    ax = plt.gca()
    plt.setp(ax.spines.values(), visible=False) # remove outer spines
    ax.tick_params(left=False, labelleft=False)
    ax.tick_params(bottom=False, labelbottom=False)
    if show:
        plt.show()
    if return_atoms:
        return atoms
