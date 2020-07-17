import json
import re
from method.atom import Atom


if __name__ == "__main__":
    with open('structure compare.json', 'r+') as fp:
        result_json = json.load(fp)
    result = []
    for element in result_json:
        element = json.loads(element)
        for key, value in element.items():
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
                element[key] = atom_list
            else:
                atom_list = []
                for atom in value:
                    atom_json = json.loads(atom)
                    atom_list.append(Atom(atom_json['pos'], type=atom_json['type'], num=atom_json['num']))
                element[key] = atom_list
        result.append(element)
    print(result)
