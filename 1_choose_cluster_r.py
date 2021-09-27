#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-

__author__ = 'Weiguo Jing'
__version__ = "2.0.0"


from method.read import poscar
from method.compare_structure import structure_compare
from method.atom import Atom
from method.plot_crystal import draw_crystal_in_ax, plot_sphere
from method.structure import Structure
import numpy as np
import matplotlib.pyplot as plt


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


def distance_of_center_and_surface(center, cell):
    norm_v_ab = np.cross(cell[0], cell[1])/np.linalg.norm(np.cross(cell[0], cell[1]))
    norm_v_bc = np.cross(cell[1], cell[2])/np.linalg.norm(np.cross(cell[1], cell[2]))
    norm_v_ca = np.cross(cell[2], cell[0])/np.linalg.norm(np.cross(cell[2], cell[0]))
    distance_ab = abs(np.dot(center, norm_v_ab))
    distance_bc = abs(np.dot(center, norm_v_bc))
    distance_ca = abs(np.dot(center, norm_v_ca))
    distance_ab = min(distance_ab, np.linalg.norm(cell[2]) - distance_ab)
    distance_bc = min(distance_bc, np.linalg.norm(cell[0]) - distance_bc)
    distance_ca = min(distance_ca, np.linalg.norm(cell[1]) - distance_ca)
    return distance_ab, distance_bc, distance_ca


def build_cluster(structure, cluster_r, doped_atom_type=None, center=None, core_r=None, tolerance=0.2):
    from method.coordination_shells import coordination_shells

    position = np.array([0.0, 0.0, 0.0])
    species = structure_species(structure)
    if doped_atom_type is None:
        doped_atom_type = species[0][0]
        print('\nTry to use *{0}* atoms as doped atoms\n'.format(doped_atom_type))
    if center is None:
        num = 0
        for atom in structure:
            if atom.type == doped_atom_type:
                position += atom.pos
                num += 1
        center = position/num
    if cluster_r > min(distance_of_center_and_surface(center, structure.cell.T)):
        print('\nERROR: the cluster is out range of the CONTCAR supercell structure. '
              'The smallest distance between center and surface is * {0} *.'
              'Please use a bigger supercell or a smaller cluster_r parameter\n'
              .format(min(distance_of_center_and_surface(center, structure.cell.T))))
        exit(1)

    cluster = Structure(structure.cell, scale=structure.scale)
    if core_r is None:
        print('\nTry to use the nearest neighbor of doped atoms to build core structure.\n')
        nearest_neighbors = []
        for atom in structure:
            if atom.type == doped_atom_type:
                neighbors = coordination_shells(structure, nshells=5, center=atom.pos, tolerance=tolerance)
                nearest_neighbors.append(atom)
                atom.pseudo = 0
                for item in neighbors[0]:
                    atom = item[0]
                    nearest_neighbors.append(atom)
                    atom.pseudo = 0
                atom = neighbors[0][0][0]
                nearest_name = atom.type
                print('\nSet *{0}* atoms as nearest neighbor.\n'.format(nearest_name))
                flag = 0
                for i in range(1, 5):
                    temp = []
                    for item in neighbors[i]:
                        if flag != 0:
                            break
                        atom = item[0]
                        if nearest_name == atom.type:
                            if not hasattr(atom, 'pseudo'):
                                temp.append(atom)
                                atom.pseudo = 0
                        else:
                            flag = 1
                    nearest_neighbors.extend(temp)
        # nearest neighbor might be outside of CONTCAR
        with open('core structure.txt', 'w+') as fp:
            for atom in nearest_neighbors:
                fp.writelines(atom.type+'   '+str(atom.pos)+'\n')
        for atom in structure:
            position = atom.pos
            d = np.linalg.norm(center - position)
            if d <= cluster_r:
                if hasattr(atom, 'pseudo'):
                    cluster.append(Atom(atom.pos - center, atom.type, pseudo=0))
                else:
                    cluster.append(Atom(atom.pos - center, atom.type, pseudo=-1))
    else:
        for atom in structure:
            d = np.linalg.norm(center - atom.pos)
            if d <= core_r:
                cluster.append(Atom(atom.pos - center, atom.type, pseudo=0))
            elif d <= cluster_r:
                cluster.append(Atom(atom.pos - center, atom.type, pseudo=-1))
    return cluster, center


def cluster_from_substrate(host, center, cluster_r):
    temp = []
    for atom in host:
        position = atom.pos
        d = np.linalg.norm(center - position)
        if d <= cluster_r:
            temp.append(atom)
    return temp


def write_compare_result_in_json(compare_result, filename='structure compare.json'):
    import re
    import json
    from copy import deepcopy

    doped_atoms_json = []
    for element in compare_result:
        element_json = deepcopy(element)
        for key, value in element.__dict__.items():
            if re.match('type', key):
                continue
            elif re.match('su', key) or re.match('un', key):
                atom_list = []
                for pair in value:
                    tmp = []
                    for atom in pair:
                        atom_json = json.dumps({'type': atom.type, 'num': atom.num, 'pos': atom.pos.tolist()})
                        tmp.append(atom_json)
                    atom_list.append(tmp)
                element_json.__dict__[key] = atom_list
            else:
                atom_list = []
                for atom in value:
                    atom_json = json.dumps({'type': atom.type, 'num': atom.num, 'pos': atom.pos.tolist()})
                    atom_list.append(atom_json)
                element_json.__dict__[key] = atom_list
        doped_atoms_json.append(json.dumps(element_json.__dict__))
    with open(filename, 'w+') as fp:
        json.dump(doped_atoms_json, fp)


if __name__ == '__main__':
    doped_crystal = 'TiAl0sc120.vasp'
    substrate = 'hostsc120.vasp'
    save = ''
    c_r = 2.5

    doped_structure = poscar(doped_crystal)
    substrate_structure = poscar(substrate)
    arrow_location, arrow, doped_atoms = structure_compare(save, substrate_structure, doped_structure,
                                                           tolerance=1, percent=1)
    write_compare_result_in_json(doped_atoms)

    doped_cluster, doped_center = build_cluster(doped_structure, c_r, tolerance=0.5)
    discard = cluster_from_substrate(substrate_structure, doped_center, c_r)
    if len(doped_cluster) != len(discard):
        print('WARNING: \n'
              'The number of atoms in cluster is different with the discard cluster of substrate.\n'
              'If there isn\'t any intersitial or vacancy in doped structure, this disagreement may caused by the '
              'choose of radius.\n'
              'The radius of cluster may be too small. Some atoms at the edge of sphere may be move out of the sphere'
              'after the relaxed doped structure\n')
        print('The number of difference is {0}'.format(len(doped_cluster) - len(discard)))

    figure = plt.figure()
    ax = figure.add_subplot(111, projection='3d')
    draw_crystal_in_ax(ax, substrate_structure, atoms_plot=False)
    plot_sphere(ax, doped_center, c_r)
    ax.quiver(arrow_location[0], arrow_location[1], arrow_location[2], arrow[0], arrow[1], arrow[2])

    tmp = []
    outer = []
    arrow_location = arrow_location.T
    arrow = arrow.T
    for i in range(len(doped_structure)):
        if np.linalg.norm(arrow_location[i]-doped_center) <= c_r:
            tmp.append(np.linalg.norm(arrow[i]))
        else:
            outer.append(np.linalg.norm(arrow[i]))
    with open('test.txt', 'w+') as fp:
        for item in tmp:
            fp.writelines('{0}\n'.format(item))
        fp.writelines('\n\n')
        for item in outer:
            fp.writelines('{0}\n'.format(item))

    plt.show()
