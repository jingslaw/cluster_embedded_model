#!/usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'jingslaw'
__version__ = 1.0

import numpy as np
from scipy import special
import time
from method.read import poscar
import spglib


def madelung_potential(cell, position, extend=(100, 100, 100), eta=10):
    volume = np.linalg.det(cell)
    reciprocal_vector1 = 2*np.pi*np.cross(cell[1], cell[2])/volume
    reciprocal_vector2 = 2*np.pi*np.cross(cell[2], cell[0])/volume
    reciprocal_vector3 = 2*np.pi*np.cross(cell[0], cell[1])/volume
    mm, nn, ll = extend[:]
    potential = 0.0
    for i in range(-mm, mm+1):
        for j in range(-nn, nn+1):
            for k in range(-ll, ll+1):
                g_vector = i*reciprocal_vector1+j*reciprocal_vector2+k*reciprocal_vector3
                vector = i*cell[0]+j*cell[1]+k*cell[2]
                if np.linalg.norm(g_vector) != 0:
                    potential += 4*np.pi*np.exp(2*np.pi*complex(0, 1)*np.dot(g_vector, position))*np.exp(-np.linalg.norm(g_vector)**2/(4*eta**2))/((np.linalg.norm(g_vector)**2)*abs(volume))
                if np.linalg.norm(vector - position) == 0:
                    potential -= 2 * eta / np.sqrt(np.pi)
                else:
                    potential += special.erfc(eta * np.linalg.norm(vector - position)) / np.linalg.norm(vector - position)
    return potential


def madelung_potential_new(cell, position, extend=(100, 100, 100), eta=10):
    fractional = np.dot(position, np.linalg.inv(cell))
    volume = np.linalg.det(cell)
    reciprocal_vector1 = 2 * np.pi * np.cross(cell[1], cell[2]) / volume
    reciprocal_vector2 = 2 * np.pi * np.cross(cell[2], cell[0]) / volume
    reciprocal_vector3 = 2 * np.pi * np.cross(cell[0], cell[1]) / volume
    mm, nn, ll = extend[:]
    x = np.linspace(-mm, mm, num=2*mm+1)
    y = np.linspace(-nn, nn, num=2*nn+1)
    z = np.linspace(-ll, ll, num=2*ll+1)
    xx, yy, zz = np.meshgrid(x, y, z, sparse=True, indexing='ij')

    coeff1 = np.exp(2*np.pi*complex(0, 1)*(xx*fractional[0]+yy*fractional[1]+zz*fractional[2]))
    coeff2 = 0.0
    coeff3 = 0.0
    for i in range(3):
        tmp = xx*reciprocal_vector1[i]+yy*reciprocal_vector2[i]+zz*reciprocal_vector3[i]
        tmp = tmp*tmp
        coeff2 += tmp
        tmp2 = (xx-fractional[0])*cell[0][i]+(yy-fractional[1])*cell[1][i]+(zz-fractional[2])*cell[2][i]
        tmp2 = tmp2*tmp2
        coeff3 += tmp2
    sum1 = 4*np.pi*coeff1*np.exp(-coeff2/(4*eta**2))/(coeff2*abs(volume))
    sum1[np.isinf(np.real(sum1))] = 0
    coeff3 = np.sqrt(coeff3)
    sum2 = special.erfc(eta*coeff3)/coeff3
    sum2[np.isinf(sum2)] = -2*eta/np.sqrt(np.pi)
    result = np.sum(sum1) + np.sum(sum2)
    if np.imag(result) > 1e-10:
        print('\nERROR: the potential {0} is not a real, unknown problem, please check the code!\n'.format(result))
        exit(1)
    return np.real(result)


if __name__ == "__main__":
    tic = time.time()
    file_name = 'YPO4primitive.vasp'
    charge = {'Y': 3, 'O': -2, 'P': 5}
    structure = poscar(file_name)
    atom_list = [[], []]
    multipler = (120, 120, 120)
    eta = 1*2*np.pi
    for atom in structure:
        atom_list[0].append(atom.pos)
        atom_list[1].append(atom.type)
    for atom in structure:
        vectors = atom.pos - np.array(atom_list[0])
        potential = 0.0
        for i in range(len(vectors)):
            potential += charge[atom_list[1][i]]*madelung_potential_new(structure.cell.T, vectors[i],
                                                                        extend=multipler, eta=eta)
        print(atom, potential)
    toc = time.time()
    print('Time:{0}'.format(toc-tic))

