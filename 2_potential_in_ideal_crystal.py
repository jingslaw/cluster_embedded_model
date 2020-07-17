#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-

__author__ = 'Weiguo Jing'
__version__ = "2.0.0"


from method.read import poscar
from method.compare_structure import compare_structure
from method.madelung import madelung_potential_new
import numpy as np


def read_compared_result_from_json(filename='structure compare.json'):
    import json
    import re
    from method.atom import Atom

    with open(filename, 'r+') as fp:
        result_json = json.load(fp)
    result = []
    for element_json in result_json:
        element_json = json.loads(element_json)
        for key, value in element_json.items():
            if re.match('type', key):
                continue
            elif re.match('su', key) or re.match('un', key):
                atom_list = []
                for pair in value:
                    tmp = []
                    for atom in pair:
                        atom_json = json.loads(atom)
                        tmp.append(Atom(atom_json['pos'], type=atom_json['type'], num=atom_json['num']))
                    atom_list.append(tmp)
                element_json[key] = atom_list
            else:
                atom_list = []
                for atom in value:
                    atom_json = json.loads(atom)
                    atom_list.append(Atom(atom_json['pos'], type=atom_json['type'], num=atom_json['num']))
                element_json[key] = atom_list
        result.append(element_json)
    return result


if __name__ == "__main__":
    doped_crystal = '2Tisc960.vasp'
    substrate = 'hostsc960.vasp'
    unit_cell_file = 'primitive.vasp'
    doped_element = 'Ti'
    already_compared_structure = True

    doped_structure = poscar(doped_crystal)
    substrate_structure = poscar(substrate)
    primitive = poscar(unit_cell_file)

    original = []
    if already_compared_structure:
        doped_atoms = read_compared_result_from_json()
        for element in doped_atoms:
            if element['type'] == doped_element:
                for pairs in element['sub']:
                    for atom in pairs:
                        if atom.type == doped_element:
                            original.append(atom)
    else:
        doped_atoms = compare_structure(substrate_structure, doped_structure, tolerance=1.0, compare_type='M')
        for element in doped_atoms:
            if element.type == doped_element:
                for pairs in element.sub:
                    for atom in pairs:
                        if atom.type == doped_element:
                            original.append(atom)

    atom_list = [[], []]
    multiplier = (100, 100, 100)
    eta = 0.1 * 2 * np.pi
    p_matrix = np.array([[0, -1, 2], [-3, 1, 1], [0, -2, 0]]) * 2
    charge = {'Al': 3, 'O': -2}

    for atom in primitive:
        atom_list[0].append(atom.pos)
        atom_list[1].append(atom.type)
    ideal_charge_potential = {}
    for atom in original:
        fraction = np.dot(np.linalg.inv(substrate_structure.cell), atom.pos)
        fraction = np.dot(fraction, p_matrix)
        fraction -= np.floor(fraction + 1e-8)
        position_in_primitive = np.dot(primitive.cell, fraction)
        vectors = position_in_primitive - np.array(atom_list[0])
        potential = 0.0
        for i in range(len(vectors)):
            if np.linalg.norm(vectors[i]) < 0.1:
                continue
            potential += charge[atom_list[1][i]] * madelung_potential_new(primitive.cell.T, vectors[i],
                                                                          extend=multiplier, eta=eta)
        ideal_charge_potential[atom.type + str(atom.num + 1)] = potential
    print(ideal_charge_potential)
