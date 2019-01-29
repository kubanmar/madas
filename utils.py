from bitarray import bitarray
import numpy as np
from ase import Atoms
import matplotlib.pyplot as plt

species_list = 'Vac,H,He,Li,Be,B,C,N,O,F,Ne,Na,Mg,Al,Si,P,S,Cl,Ar,K,Ca,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Ga,Ge,As,Se,Br,Kr,Rb,Sr,Y,Zr,Nb,Mo,Tc,Ru,Rh,Pd,Ag,Cd,In,Sn,Sb,Te,I,Xe,Cs,Ba,La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi,Po,At,Rn,Fr,Ra,Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr,Rf,Db,Sg,Bh,Hs,Mt,Ds,Rg,Cn,Nh,Fl,Mc,Lv,Ts,Og'.split(',')
electron_charge = 1.602176565e-19

def _SI_to_Angstom(length):
    return np.power(length,10^10)

def plot_FP_in_grid(byte_fingerprint, grid_id): #TODO adapt to current code version
    x=[]
    y=[]
    all_width=[]
    bin_fp=bitarray()
    bin_fp.frombytes(byte_fingerprint.bins)
    grid_indices=byte_fingerprint.indices
    plotgrid=grid.object.grid()
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
    plt.bar(x,y,width=all_width,align='edge')

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
        dos.append(dos_json['dos_values'][0][index]/ 1e-30 /electron_charge/volume)
    return name,energy,dos

def plot_similar_dos(reference_mid, sim_dict, database, show = True, nmax = 10):
    """
    Plot DOS of materials similar to a reference material.
    """
    plt.figure()
    label, energy, dos = get_plotting_data(reference_mid, database)
    label=str(label)+' (reference)'
    plt.plot(energy,dos,label=label,alpha=0.5)
    plt.fill_between(energy,0,dos,alpha=0.5)
    similar_materials = [(sim_dict[reference_mid][key], key) for key in sim_dict[reference_mid].keys()]
    similar_materials.sort(reverse = True)
    for index, item in enumerate(similar_materials):
        mid = item[1]
        label, energy, dos = get_plotting_data(mid, database)
        label = label + ' S='+str(round(float(item[0]), 5))
        if index < nmax:
            plt.plot(energy,dos,label=label,alpha=0.5)
            plt.fill_between(energy,0,dos,alpha=0.5/(index+2))
    plt.axis(fontsize='25')
    plt.xlabel('Energy [eV]',fontsize='40')
    plt.ylabel('DOS [states/unit cell/eV]',fontsize='40')
    plt.xticks(fontsize='30')
    plt.yticks(fontsize='30')
    plt.legend(fontsize='10')
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
