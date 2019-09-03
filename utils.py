from bitarray import bitarray
import numpy as np
from ase import Atoms
import matplotlib.pyplot as plt

species_list = 'Vac,H,He,Li,Be,B,C,N,O,F,Ne,Na,Mg,Al,Si,P,S,Cl,Ar,K,Ca,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Ga,Ge,As,Se,Br,Kr,Rb,Sr,Y,Zr,Nb,Mo,Tc,Ru,Rh,Pd,Ag,Cd,In,Sn,Sb,Te,I,Xe,Cs,Ba,La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi,Po,At,Rn,Fr,Ra,Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr,Rf,Db,Sg,Bh,Hs,Mt,Ds,Rg,Cn,Nh,Fl,Mc,Lv,Ts,Og'.split(',')
electron_charge = 1.602176565e-19

def rmsle(y_true, y_pred):
    if not isinstance(y_true, (list, np.ndarray)):
        y_true = [y_true]
    if not isinstance(y_pred, (list, np.ndarray)):
        y_pred = [y_pred]
    errors = [np.log((np.array(y_p) + 1)/(np.array(y_t) + 1))**2 for y_t, y_p in zip(y_true, y_pred)]
    msle = sum(errors) / len(errors)
    return np.sqrt(msle)

def _SI_to_Angstom(length):
    return np.power(length,10^10)

def report_error(logger, error_message):
    if logger != None:
        logger.error(error_message)
    else:
        print(error_message)

def plot_FP_in_grid(byte_fingerprint, grid, show_grid = True, show = True, label = '', axes = None, **kwargs):
    x=[]
    y=[]
    all_width=[]
    bin_fp=bitarray()
    bin_fp.frombytes(byte_fingerprint.bins)
    grid_indices=byte_fingerprint.indices
    plotgrid=grid.grid()
    plotgrid=plotgrid[grid_indices[0]:grid_indices[1]]
    gridded_fp_energy=[x[0] for x in plotgrid]
    bit_position=0
    for index,item in enumerate(plotgrid):
        if index<len(plotgrid)-1:
            width=plotgrid[index+1][0]-item[0]
        else:
            width=abs(item[0]-plotgrid[index-1][0])
        for dos_value in item[1]:
            if bin_fp[bit_position]==1:
                x.append(item[0])
                y.append(dos_value)
                all_width.append(width)
            bit_position+=1
    if axes == None:
        plt.bar(x,y,width=all_width,align='edge', label = label)
    else:
        axes.bar(x,y,width=all_width,align='edge', label = label)
    if show_grid:
        grid.plot_grid_bars(show = False, axes = axes, **kwargs)
    if show:
        plt.show()

def get_plotting_data(mid, database):
    electron_charge=1.6021766e-19
    name = database.get_formula(mid)
    dos_json = database.get_property(mid, 'dos')
    volume = database.get_property(mid, 'cell_volume')
    energy=[]
    dos=[]
    for index in range(len(dos_json['dos_energies'])):
        energy.append(dos_json['dos_energies'][index]/electron_charge)
        #print(dos_json['dos_values'][index][0])
        dos.append(dos_json['dos_values'][0][index]/ 1e-30 /electron_charge) #WARNING only using current data
    return name,energy,dos

def sub_numbers(string):
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
    return  r'$\mathrm{'+subbed_string+r'}$'

def plot_similar_dos(reference_mid, sim_dict, database, show = True, nmax = 10, figsize = (5,5)):
    """
    Plot DOS of materials similar to a reference material.
    """
    plt.figure(figsize = figsize)
    label, energy, dos = get_plotting_data(reference_mid, database)
    label=sub_numbers(label)+' (reference)'
    plt.plot(energy,dos,label=label,alpha=0.5)
    plt.fill_between(energy,0,dos,alpha=0.5)
    similar_materials = [(sim, mid) for mid, sim in sim_dict[reference_mid]]
    similar_materials.sort(reverse = True)
    for index, item in enumerate(similar_materials):
        mid = item[1]
        label, energy, dos = get_plotting_data(mid, database)
        label =sub_numbers(label) + r' Tc=' +str(round(float(item[0]), 5))
        if index < nmax and mid != reference_mid:
            plt.plot(energy,dos,label=label,alpha=0.5)
            plt.fill_between(energy,0,dos,alpha=0.5/(index+2))
        if mid == reference_mid:
            nmax += 1
    plt.axis(fontsize='25')
    plt.xlabel('Energy [eV]',fontsize='40')
    plt.ylabel('DOS [states/unit cell/eV]',fontsize='40')
    plt.xticks(fontsize='30')
    plt.yticks(fontsize='30')
    plt.legend(fontsize='20')
    if show:
        plt.show()

def plot_dos_material_list(material_list, db, show = False, figsize = (10,10)):
    plt.figure()
    for index, material in enumerate(material_list):
        name, energy, dos = get_plotting_data(material, db)
        plt.plot(energy, dos, label = name)
        plt.legend()
    if show:
        plt.show()

def plot3D(x,y,z, show = True, xlabel = 'X', ylabel = 'Y', zlabel = 'Z'):
    from matplotlib import cm, colors
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z, depthshade=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if show:
        plt.show()

def plot_contour(z_matrix, xvals, yvals, show = True, xlabel = 'X', ylabel = 'Y', zlabel = 'Z'):
    fig1, ax2 = plt.subplots(constrained_layout=True)
    cs = ax2.contourf(xvals, yvals, z_matrix)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    cbar = fig1.colorbar(cs)
    cbar.ax.set_ylabel(zlabel)
    if show:
        plt.show()

def get_lattice_parameters_from_string(string):
    lcs=string.split(',')
    lcs[0]=lcs[0][1:]
    lcs[-1]=lcs[-1][:-1]
    lcs_float=[]
    for index,item in enumerate(lcs):
        if index<3:
            lcs_float.append(float(item)*10**(10))
        else:
            lcs_float.append(float(item)/np.pi*180)
    return lcs_float

def get_lattice_description(elements, lattice_parameters):
    labels=[species_list[x[0]] for x in elements]
    positions = [x[1] for x in elements]
    cell=get_lattice_parameters_from_string(lattice_parameters)
    scaled_positions=[[x[0] * cell[0], x[1] * cell[1], x[2] * cell[2]] for x in positions]
    structure=Atoms(symbols=labels,positions=scaled_positions,cell=cell, pbc = True)
    return structure

def list_chunks(long_list, chunk_length = 5):
    chunked_list = []
    chunk = []
    for index,item in enumerate(long_list):
        chunk.append(item)
        if index % chunk_length == 0:
            chunked_list.append(chunk)
            chunk = []
    if chunk != []:
        chunked_list.append(chunk)
    return chunked_list

import json,sys
from os.path import exists

def _seek_terminating_char(f, char = '}'):
    for offset in range(1,11):
        byteoffset = offset * (-1)
        f.seek(byteoffset, 2)
        data = f.read()
        if bytes(char,'utf-8') in data:
            f.seek(byteoffset, 2)
            return
    sys.exit("No valid char found.")

def write_json_file(json_data, filename):
    if not exists(filename):
        with open(filename,'w') as f:
            json.dump(json_data, f, indent = 4)
    else:
        with open(filename, 'rb+') as f:
            _seek_terminating_char(f)
            f.write(bytes(',','utf-8'))
            f.write(bytes(json.dumps(json_data, indent = 4),'utf-8')[1:])

def merge_k_nearest_neighbor_dicts(fp_type_list, dicts):
    merged_dict = {}
    for index, fp_type in enumerate(fp_type_list):
        materials = [key for key in dicts[index].keys()]
        for material in materials:
            data = dicts[index][material]
            if not material in merged_dict.keys():
                merged_dict[material] = {mid:{fp_type:sim} for mid, sim in data}
            else:
                for mid, sim in data:
                    if not mid in merged_dict[material].keys():
                        merged_dict[material][mid] = {fp_type:sim}
                    else:
                        merged_dict[material][mid].update({fp_type:sim})
    return merged_dict
