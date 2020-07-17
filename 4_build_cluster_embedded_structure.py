#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-

__author__ = 'Weiguo Jing'
__version__ = "2.0.0"


from method.read import poscar
from method.structure import Structure
from method.atom import Atom
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


def distance_of_center_and_surface(center, cell):
    norm_v_ab = np.cross(cell[0], cell[1])/np.linalg.norm(np.cross(cell[0], cell[1]))
    norm_v_bc = np.cross(cell[1], cell[2])/np.linalg.norm(np.cross(cell[1], cell[2]))
    norm_v_ca = np.cross(cell[2], cell[0])/np.linalg.norm(np.cross(cell[2], cell[0]))
    distance_ab = abs(np.dot(center, norm_v_ab))
    distance_bc = abs(np.dot(center, norm_v_bc))
    distance_ca = abs(np.dot(center, norm_v_ca))
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
              'Please use a bigger supercell or a smaller cluster_r parameter\n')
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
                    if flag != 0:
                        break
                    for item in neighbors[i]:
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


def extend_structure(structure, shell_r):
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


def build_shell(structure, center, cluster_r, shell_r, pure_substrate=False, remove_dict=None):

    if pure_substrate is False:
        if remove_dict is None:
            print('ERROR: remove_list can not be None when substrate includes doped atoms.\n')
            exit(1)
        structure = remove_doped_atoms(structure, remove_dict)
    supercell = extend_structure(structure, shell_r)
    shell = Structure(structure.cell, scale=structure.scale)
    temp = []
    for atom in supercell:
        position = atom.pos
        d = np.linalg.norm(position - center)
        if cluster_r < d <= shell_r:
            temp.append(Atom(position - center, atom.type))
    shell.extend(temp)
    return shell


def build_mosaic_structure(cluster, shell):
    for atom in shell:
        cluster.append(Atom(atom.pos, atom.type, pseudo=-1))
    return cluster


def write_seward_input(structure, core_pseudo, mosaic_pseudo, charge, mosaic_r, file_name='1_sew.in', title=None):
    core = {}
    mosaic = {}
    count = 0
    for key in core_pseudo:
        core[key] = []
    for key in mosaic_pseudo:
        mosaic[key] = []
    ion = []
    for atom in structure:
        if atom.pseudo == 0:
            try:
                core[atom.type].append(atom)
            except KeyError:
                print('ERROR: Core pseudo potential do not include the information about {0}\n'.format(atom.type))
                exit(1)
            finally:
                count += 1
        elif np.linalg.norm(atom.pos) <= mosaic_r:
            try:
                mosaic[atom.type].append(atom)
            except KeyError:
                print('ERROR: mosaic pseudo potential do not include the information about {0}\n'.format(atom.type))
                exit(1)
            finally:
                count += 1
        else:
            ion.append(atom)
    string = " &SEWARD &END\nTitle\n{0}\n".format(title)
    for key in core.keys():
        temp = "Basis set\n{0}\n".format(core_pseudo[key])
        i = 1
        for atom in core[key]:
            temp = temp + "{name}{num}{pos0}{pos1}{pos2}  Angstrom \n".format(name=atom.type, num='{:<6d}'.format(i),
                                                                              pos0='{:10.4f}'.format(atom.pos[0]),
                                                                              pos1='{:10.4f}'.format(atom.pos[1]),
                                                                              pos2='{:10.4f}'.format(atom.pos[2]))
            i += 1
        temp = temp + "End of basis\n********************************************\n"
        string = string + temp
    abc_code = 65
    for key in mosaic.keys():
        temp = "Basis set\n{0}\n".format(mosaic_pseudo[key])
        i = 1
        for atom in mosaic[key]:
            temp = temp + "{char}{num}{pos0}{pos1}{pos2}  Angstrom \n".format(char=chr(abc_code),
                                                                              num='{:<6d}'.format(i),
                                                                              pos0='{:10.4f}'.format(atom.pos[0]),
                                                                              pos1='{:10.4f}'.format(atom.pos[1]),
                                                                              pos2='{:10.4f}'.format(atom.pos[2]))
            i += 1
        temp = temp + "End of basis\n********************************************\n"
        string = string + temp
        if i >= 1000:
            print('WARNING: There are too many {0} atoms in mosaic shell, which cause the label of atoms at first '
                  'column has more than 4 characters.\nIt will cause Molcas reduce the label to 4 characters and '
                  'cause *reduplicate* problem. e.g. "A1000" will be reduced to "A100" in Molcas.\n'.format(key))
        abc_code = (abc_code - 65 + 1) % 26 + 65
        if (abc_code - 65 + 1) // 26 > 1:
            print('WARNING: Too many species in mosaic shell. The label of atoms at first column will be '
                  '*reduplicate*.\n')
    string = string + "Xfield\n{0}  Angstrom\n".format(len(ion))
    for atom in ion:
        q = 0.0
        try:
            q = charge[atom.type]
        except KeyError:
            print('ERROR: The charge information about {0} is needed\n'.format(atom.type))
            exit(1)
        string = string + "{pos0}{pos1}{pos2}{charge}  0.0  0.0  0.0\n".format(pos0='{:10.4f}'.format(atom.pos[0]),
                                                                               pos1='{:10.4f}'.format(atom.pos[1]),
                                                                               pos2='{:10.4f}'.format(atom.pos[2]),
                                                                               charge='{:10.4f}'.format(q))
    string = string + 'AMFI\nSDIPOLE\nEnd of input \n'
    if count > 500:
        print('\nWARNING: The default max active atoms number in MOLCAS is 500, you have {0} active atoms in core'
              'and mosaic structure. This may cause ERROR *RdCtl: Increase Mxdc* in sew calculation\n'.format(count))
    with open(file_name, 'w') as fp:
        fp.write(string)


if __name__ == '__main__':
    doped_crystal = '2Tisc960.vasp'
    substrate = 'hostsc960.vasp'
    c_r = 6
    s_r = 16
    m_r = 9
    doped_structure = poscar(doped_crystal)
    substrate_structure = poscar(substrate)
    doped_cluster, doped_center = build_cluster(doped_structure, c_r, tolerance=0.5)
    discard = cluster_from_substrate(substrate_structure, doped_center, c_r)
    if len(doped_cluster) != len(discard):
        print('WARNING: \n'
              'The number of atoms in cluster is different with the discard cluster of substrate.\n'
              'If there isn\'t any intersitial or vacancy in doped structure, this disagreement may caused by the '
              'choose of radius.\n'
              'The radius of cluster may be too small. Some atoms at the edge of sphere may be move out of the sphere'
              'after the relaxed doped structure\n')
    shell = build_shell(substrate_structure, doped_center, c_r, s_r, pure_substrate=False, remove_dict={'Ti': 'Al'})
    cluster_embedded_structure = build_mosaic_structure(doped_cluster, shell)

    core_pseudo = {'Ti': 'Ti.ECP.Barandiaran.9s6p6d3f.3s3p4d1f.10e-CG-AIMP.',
                   'O': 'O.ECP.Barandiaran.5s6p1d.2s4p1d.6e-CG-AIMP.'}
    mosaic_pseudo = {'Al': 'Al.ECP.Pascual.0s.0s.0e-AIMP-Al2O3.',
                     'O': 'O.ECP.Pascual.0s.0s.0e-AIMP-Al2O3.'}
    charge = {'Al': 3, 'O': -2}
    write_seward_input(cluster_embedded_structure, core_pseudo, mosaic_pseudo, charge, m_r, title='Al2O3_Ti')
