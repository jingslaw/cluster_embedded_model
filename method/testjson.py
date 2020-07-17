from method.read import poscar
from method.compare_structure import compare_structure
from copy import deepcopy
import json
import re

if __name__ == "__main__":
    doped_crystal = '2Tisc960.vasp'
    substrate = 'hostsc960.vasp'

    doped_structure = poscar(doped_crystal)
    substrate_structure = poscar(substrate)
    doped_atoms = compare_structure(substrate_structure, doped_structure, tolerance=1.0, compare_type='Q')
    doped_atoms_json = []
    for element in doped_atoms:
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
    with open('structure compare.json', 'w+') as fp:
        json.dump(doped_atoms_json, fp)
