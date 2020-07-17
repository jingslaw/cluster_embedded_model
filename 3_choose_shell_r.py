#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-

__author__ = 'Weiguo Jing'
__version__ = "2.0.0"


from method.read import poscar
from method.atom import Atom
from method.structure import Structure
import numpy as np


def structure_species(structure):
    from method.defect import specieset
    atoms_type = specieset(structure)
    species = {}
    for name in atoms_type:
        count = 0
        for atom in structure:
            if atom.type == name:
                count += 1
        species[name] = count
    return sorted(species.items(), key=lambda item: item[1])


def remove_doped_atoms(structure, remove_dict):
    temp = []
    for atom in structure:
        if atom.type in remove_dict.keys():
            value = remove_dict[atom.type]
            if value != '':
                temp.append(Atom(atom.pos, value))
        else:
            temp.append(Atom(atom.pos, atom.type))
    new_structure = Structure(structure.cell, scale=structure.scale)
    new_structure.extend(temp)
    return new_structure


def extend_structure(structure, shell_r, pure_substrate=True, remove_dict=None):
    if pure_substrate is False:
        if remove_dict is None:
            print('ERROR: remove_list can not be None when substrate includes doped atoms.\n')
            exit(1)
        structure = remove_doped_atoms(structure, remove_dict)
    cell = structure.cell.T
    volume = structure.volume
    hz = volume/np.linalg.norm(np.cross(cell[0], cell[1]))
    hx = volume/np.linalg.norm(np.cross(cell[1], cell[2]))
    hy = volume/np.linalg.norm(np.cross(cell[2], cell[0]))
    if min(hx, hy, hz)/2 >= shell_r:
        return structure
    nz = int(np.ceil(shell_r/hz - 0.5))
    nx = int(np.ceil(shell_r/hx - 0.5))
    ny = int(np.ceil(shell_r/hy - 0.5))
    temp = []
    for i in range(-nx, nx+1):
        for j in range(-ny, ny+1):
            for k in range(-nz, nz+1):
                for atom in structure:
                    position = atom.pos + i*cell[0] + j*cell[1] + k*cell[2]
                    temp.append(Atom(position, atom.type))
    supercell = Structure(structure.cell, scale=structure.scale)
    supercell.extend(temp)
    return supercell


def get_onion_structure_from_supercell(supercell, center, shell_range, number_of_shells=-1):
    onion = [[], ]
    onion_core = 0
    end = shell_range[1]
    start = shell_range[0]
    if number_of_shells <= 0:
        step = 1
        number_of_shells = np.ceil(end - start)
    else:
        step = (end - start) / number_of_shells
    for i in range(int(number_of_shells)):
        onion.append([])
    for atom in supercell:
        distance = np.linalg.norm(atom.pos - center)
        if distance <= start:
            onion[onion_core].append(atom)
        elif distance > end:
            continue
        else:
            shells_no = np.ceil((distance - start) / step)
            onion[int(shells_no)].append(atom)
    return onion


def the_potential_from_onion(location, onion, charge, distance_tolerance=0.1):
    potential_in_onion = []
    for onion_shell in onion:
        potential = 0.0
        for atom in onion_shell:
            distance = np.linalg.norm(atom.pos - location)
            if distance < distance_tolerance:
                continue
            else:
                potential += charge[atom.type] / distance
        potential_in_onion.append(potential)
    for i in range(len(potential_in_onion)-1):
        potential_in_onion[i+1] = potential_in_onion[i+1] + potential_in_onion[i]
    return potential_in_onion


if __name__ == "__main__":
    doped_crystal = '2Tisc960.vasp'
    substrate = 'hostsc960.vasp'
    doped_atom_type = None
    doped_center = None
    max_shell_r = 60
    cluster_r = 9
    q = {'Al': 3, 'O': -2}

    doped_structure = poscar(doped_crystal)
    substrate_structure = poscar(substrate)

    if doped_atom_type is None:
        species = structure_species(doped_structure)
        doped_atom_type = species[0][0]
        print('\nTry to use *{0}* atoms as doped atoms\n'.format(doped_atom_type))
    if doped_center is None:
        position = np.array([0.0, 0.0, 0.0])
        num = 0
        for atom in doped_structure:
            if atom.type == doped_atom_type:
                position += atom.pos
                num += 1
        doped_center = position / num
    doped_atoms = []
    for atom in doped_structure:
        if atom.type == doped_atom_type:
            doped_atoms.append(atom)
    supercell = extend_structure(substrate_structure, max_shell_r, pure_substrate=True, remove_dict=None)
    point_charge_structure = get_onion_structure_from_supercell(supercell, doped_center, [cluster_r, max_shell_r])
    potential_result = []
    for atom in doped_atoms:
        potentials = the_potential_from_onion(atom.pos, point_charge_structure, q, distance_tolerance=0.1)
        potential_result.append(potentials)
    with open('the potential in defect site.txt', 'w+') as fp:
        for i in range(len(doped_atoms)):
            atom = doped_atoms[i]
            potential_in_shells = potential_result[i]
            fp.writelines('{0}{1}\n'.format(atom.type, i))
            fp.writelines('position:{0}\n\n'.format(atom.pos - doped_center))
            fp.writelines('{0}      {1}     \n'.format('No.', 'potential'))
            for j in range(len(potential_in_shells)):
                fp.writelines('{0}      {1}     \n'.format(j, potential_in_shells[j]))
            fp.writelines('\n')
