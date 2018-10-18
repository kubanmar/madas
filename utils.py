from bitarray import bitarray
import numpy as np
from ase import Atoms

species_list = 'Vac,H,He,Li,Be,B,C,N,O,F,Ne,Na,Mg,Al,Si,P,S,Cl,Ar,K,Ca,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Ga,Ge,As,Se,Br,Kr,Rb,Sr,Y,Zr,Nb,Mo,Tc,Ru,Rh,Pd,Ag,Cd,In,Sn,Sb,Te,I,Xe,Cs,Ba,La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi,Po,At,Rn,Fr,Ra,Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr,Rf,Db,Sg,Bh,Hs,Mt,Ds,Rg,Cn,Nh,Fl,Mc,Lv,Ts,Og'.split(',')
electron_charge = 1.602176565e-19

def plot_FP_in_grid(byte_fingerprint, grid_id):
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
    ppl.bar(x,y,width=all_width,align='edge')

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
