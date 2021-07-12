import itertools
import math
from typing import List

import numpy as np
from fa2 import ForceAtlas2


def truncate(f, n):
    """Truncates/pads a float f to n decimal places without rounding"""
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d + '0' * n)[:n]])


class State:

    def __init__(self, name, transitions=None, start=False, end=False):
        self.name = name
        self.transitions = transitions
        self.start = start
        self.end = end
        self.visited = False

    def __eq__(self, other):
        return self.name == other.name

    def __gt__(self, other):
        return self.name > other.nama

    def __lt__(self, other):
        return self.name < other.name

    def __repr__(self):
        return f"{self.name}"

    def latex(self, x, y):
        x, y = truncate(x, 4), truncate(y, 4)
        name = self.name.replace(" ", "").replace(",", "")
        if self.start and self.end:
            return f"\\node[state, initial, accepting] at ({x}, {y}) ({name}) {'{' + self.name + '}'};"
        if self.start:
            return f"\\node[state, initial] at ({x}, {y}) ({name}) {'{' + self.name + '}'};"
        if self.end:
            return f"\\node[state, accepting] at ({x}, {y}) ({name}) {'{' + self.name + '}'};"
        return f"\\node[state] at ({x}, {y}) ({name}) {'{' + self.name + '}'};"

    def path(self):
        if not self.visited:
            self.visited = True
            if self.transitions and self.end:
                return f"({self.name}, is_end: {self.end}) " + str([str(x) for x in self.transitions]) \
                    .replace("'", "").replace('"', "")
            if self.transitions and self.start:
                return f"({self.name}, is_start: {self.start}) " + str([str(x) for x in self.transitions]) \
                    .replace("'", "").replace('"', "")
            if self.transitions and self.start and self.end:
                return f"({self.name}, is_start: {self.start}. is_end: {self.end}) " + str(
                    [str(x) for x in self.transitions]).replace("'", "").replace('"', "")
            if self.transitions:
                return f"({self.name}) " + str([str(x) for x in self.transitions]).replace("'", "").replace('"', "")
            if self.start:
                return f"({self.name}, is_start: {self.start})"
            if self.end:
                return f"({self.name} is_end: {self.end})"
            return f"({self.name})"
        return f"({self.name})"


class Transition:

    def __init__(self, end_state: State, requirement):
        self.end_state = end_state
        self.requirement = requirement

    def __repr__(self):
        return f"-{self.requirement}->{self.end_state}"

    def latex(self, _start_state, position):
        """:param position [loop, above, below, bend, left, right]"""
        return f"\\draw ({_start_state.name.replace(' ', '').replace(',', '')}) edge[{position}] node " \
               f"{'{' + self.requirement + '}'} ({self.end_state.name.replace(' ', '').replace(',', '')});"


def power_set_construction(_states: List[State], _transitions: str):
    states_power_set = [x for length in range(len(_states) + 1) for x in itertools.combinations(_states, length)]
    _result = []
    for p_states in states_power_set:
        if p_states:
            cmp_sets = []
            for transition in _transitions:
                res = list()
                for state in p_states:
                    for t in state.transitions:
                        if t.requirement == transition:
                            if t.end_state not in res:
                                res.append(t.end_state)
                cmp_sets.append(sorted(res, key=lambda s: s.name))
            if cmp_sets[0] == cmp_sets[1] and cmp_sets[0]:
                print(f"f({list(p_states)}, {_transitions}) = {cmp_sets[0]}")
                _result.append(((list(p_states), _transitions), cmp_sets[0]))
            else:
                if cmp_sets[0]:
                    print(f"f({list(p_states)}, {_transitions[0]}) = {cmp_sets[0]}")
                    _result.append(((list(p_states), _transitions[0]), cmp_sets[0]))
                if cmp_sets[1]:
                    print(f"f({list(p_states)}, {_transitions[1]}) = {cmp_sets[1]}")
                    _result.append(((list(p_states), _transitions[1]), cmp_sets[1]))
    return _result


def check_for_incoming_transitions(_states, name):
    for _name, _value in _states.items():
        if _name != name:
            if any([name == _transition.end_state.name for _transition in _value.transitions]):
                return True
    return False


def nfs_power_set_to_dfa(power_set_result):
    _states = {}
    for value in power_set_result:
        v1, _end_state = value
        _start_state, transition = v1

        str_start_state = str(_start_state).replace(",", "").replace("[", "").replace("]", "")  # .replace(" ", "")
        str_end_state = str(_end_state).replace(",", "").replace("[", "").replace("]", "")  # .replace(" ", "")

        if str_start_state not in _states.keys():
            _states[str_start_state] = State(str_start_state, [], all([x.start for x in _start_state]),
                                             any([x.end for x in _start_state]))
        if str_end_state not in _states.keys():
            _states[str_end_state] = State(str_end_state, [], all([x.start for x in _end_state]),
                                           any([x.end for x in _end_state]))

    for value in power_set_result:
        v1, _end_state = value
        _start_state, transition = v1

        str_start_state = str(_start_state).replace(",", "").replace("[", "").replace("]", "")  # .replace(" ", "")
        str_end_state = str(_end_state).replace(",", "").replace("[", "").replace("]", "")  # .replace(" ", "")
        _states.get(str_start_state).transitions.append(Transition(_states.get(str_end_state), transition))

    ret_states = _states.copy()
    for name in _states.keys():
        if not check_for_incoming_transitions(_states, name) and not _states.get(name).start:
            ret_states.pop(name)
    _states = ret_states.copy()
    for name in _states.keys():
        if not check_for_incoming_transitions(_states, name) and not _states.get(name).start:
            ret_states.pop(name)

    _matrix = np.zeros((len(ret_states), len(ret_states)))

    # for _y in range(len(ret_states)):
    #    for _x in range(len(ret_states)):
    #        _matrix[_y][_x] = -10

    _name_to_index, _index_to_name = {}, {}
    _index = 0
    for name in ret_states.keys():
        _name_to_index[name] = _index
        _index_to_name[_index] = name
        _index += 1

    for name, value in ret_states.items():
        _y = _name_to_index[name]

        multiplier = 1

        for _name, _value in ret_states.items():
            if _name != name:
                for _transition in _value.transitions:
                    if _transition.end_state.name == value.name:
                        multiplier -= 1
        # multiplier += 10 if value.start else 0
        # multiplier += 5 if value.end else 0

        for transition in value.transitions:
            _x = _name_to_index[transition.end_state.name]
            _matrix[_y][_x] = 1 * (len(value.transitions) + multiplier)
            _matrix[_x][_y] = 1 * (len(value.transitions) + multiplier)

    return ret_states, _matrix, _index_to_name, _name_to_index


def nfa_to_matrix(states):
    _states = {x.name: x for x in states}

    _matrix = np.zeros((len(_states), len(_states)))
    _name_to_index, _index_to_name = {}, {}
    _index = 0
    for name in _states.keys():
        _name_to_index[name] = _index
        _index_to_name[_index] = name
        _index += 1

    for name, value in _states.items():
        _y = _name_to_index[name]

        multiplier = 1

        for _name, _value in _states.items():
            if _name != name:
                for _transition in _value.transitions:
                    if _transition.end_state.name == value.name:
                        multiplier += 2
        multiplier += 10 if value.start else 0
        multiplier += 5 if value.end else 0

        for transition in value.transitions:
            _x = _name_to_index[transition.end_state.name]
            _matrix[_y][_x] = (len(value.transitions) - len(value.name) * 0.5) * multiplier
            _matrix[_x][_y] = (len(value.transitions) - len(value.name) * 0.5) * multiplier
    return _states, _matrix, _index_to_name, _name_to_index


state_z0 = State("S0", [], start=True)
state_z4 = State("S1", [], start=True)
state_z1 = State("A", [])
state_z2 = State("B", [])
state_z3 = State("F", [], end=True)

state_z0.transitions = [Transition(state_z1, 'a')]
state_z4.transitions = [Transition(state_z2, 'b')]

state_z1.transitions = [Transition(state_z3, 'a')]
state_z2.transitions = [Transition(state_z3, 'b')]

state_z3.transitions = [Transition(state_z3, 'a'),
                        Transition(state_z3, 'b')]

# print(state_z0.path())
print('hallo Lukas :)')
states = [state_z0, state_z1, state_z2, state_z3]
transitions = "ab"
result = power_set_construction(states, transitions)


def start_positions(n, size, r):
    _positions = np.zeros((size, 2))
    _circ_center_y = _circ_center_x = size / 2
    _index = 0

    _positions[0][0] = _circ_center_x
    _positions[0][1] = _circ_center_y

    for i in range(n - 1):
        y = _circ_center_y + r * math.sin(_index)
        x = _circ_center_x + r * math.cos(_index)
        _index += size
        if _index >= 360 - (size * 2):
            _index = 0
            r += 2
        _positions[i + 1][0] = x
        _positions[i + 1][1] = y
    return _positions


# pprint(result)

new_states, matrix, index_to_name, name_to_index = nfs_power_set_to_dfa(result)

forceatlas2 = ForceAtlas2(
    # Behavior alternatives
    outboundAttractionDistribution=True,  # Dissuade hubs
    linLogMode=False,  # NOT IMPLEMENTED
    adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
    edgeWeightInfluence=1.0,

    # Performance
    jitterTolerance=1.0,  # Tolerance
    barnesHutOptimize=True,
    barnesHutTheta=1.2,
    multiThreaded=False,  # NOT IMPLEMENTED

    # Tuning
    scalingRatio=2.0,
    strongGravityMode=False,
    gravity=1.0,

    # Log
    verbose=True)

"""state_z0 = State("Z0", [], start=True)
state_z1 = State("Z1", [])
state_z2 = State("Z2", [])
state_z3 = State("Z3", [])
state_z4 = State("Z4", [])
state_z5 = State("Z5", [], end=True)

state_z0.transitions = [Transition(state_z1, 'm')]

state_z1.transitions = [Transition(state_z2, 'm')]

state_z2.transitions = [Transition(state_z3, 'm')]

state_z3.transitions = [Transition(state_z4, 'm'), Transition(state_z5, 'm'), Transition(state_z0, 'm')]
state_z4.transitions = [Transition(state_z5, 'm'), Transition(state_z0, 'm')]

states = [state_z0, state_z1, state_z2, state_z3, state_z4, state_z5]

new_states, matrix, index_to_name, name_to_index = nfa_to_matrix(states)

state_z0 = State("Z0", [], start=True)
state_z1 = State("Z1", [])
state_z2 = State("Z2", [])
state_z3 = State("Z3", [], end=True)
state_z4 = State("Z4", [], start=True)
state_z5 = State("Z5", [])
state_z6 = State("Z6", [])
state_z7 = State("Z7", [])
state_z8 = State("Z8", [], start=True)
state_z9 = State("Z9", [])
state_z10 = State("Z10", [])
state_z11 = State("Z11", [])
state_z12 = State("Z12", [])

state_z0.transitions = [Transition(state_z1, '1, ..., 9')]
state_z1.transitions = [Transition(state_z2, '2, ..., 9')]
state_z2.transitions = [Transition(state_z3, '4, ..., 9')]

state_z4.transitions = [Transition(state_z5, '1, ..., 9')]
state_z5.transitions = [Transition(state_z6, '0, ..., 9')]
state_z6.transitions = [Transition(state_z7, '0, ..., 9')]
state_z7.transitions = [Transition(state_z3, '0, ..., 9')]

state_z8.transitions = [Transition(state_z9, '1, ..., 9')]
state_z9.transitions = [Transition(state_z10, '0, ..., 9')]
state_z10.transitions = [Transition(state_z11, '0, ..., 9')]
state_z11.transitions = [Transition(state_z12, '0, ..., 9'), Transition(state_z3, '0, ..., 9')]
state_z12.transitions = [Transition(state_z3, '0, ..., 9')]

states = [state_z0, state_z1, state_z2, state_z3, state_z4, state_z5, state_z6, state_z7, state_z8, state_z9, state_z10,
          state_z11, state_z12]

new_states, matrix, index_to_name, name_to_index = nfa_to_matrix(states)
"""

positions = forceatlas2.forceatlas2(matrix, pos=None, iterations=4)  # start_positions(len(new_states), 45, 2)

print(positions)
print(len(positions))

_latex_nodes = ""
_latex_transitions = ""
top = "above"

MULTIPLIER = 0.50  # 0.4
FILE_NAME = "__nfa"
file_name = FILE_NAME
ROTATE_X_Y = False
X_SHIFT = 10

for index in range(len(new_states)):

    x, y = positions[index]
    name = index_to_name[index]
    state = new_states[name]

    if ROTATE_X_Y:
        _latex_nodes += state.latex((y + X_SHIFT) * MULTIPLIER, x * MULTIPLIER) + "\n"
    else:
        _latex_nodes += state.latex((x + X_SHIFT) * MULTIPLIER, y * MULTIPLIER) + "\n"

    for _transition in state.transitions:
        print(f"{state.name}: {_transition}")
        if _transition.end_state == state:
            _latex_transitions += _transition.latex(state, "loop, above") + "\n"
        else:
            _latex_transitions += _transition.latex(state, f"bend right, {top}") + "\n"
            top = "below" if top == "above" else top

print(_latex_nodes)
print(_latex_transitions)

path = "C:\\Users\\Lukas\\Dropbox\\Informatik\\so2021\\Formale Sprachen und Komplexität\\04"

_latex_document_start = """
\\documentclass[a4paper,oneside, 12pt]{article}
\\usepackage{tikz}
\\usetikzlibrary{automata, positioning, arrows}
\\tikzset{
->,
>=stealth',
node distance=3cm, % specifies the minimum distance between two nodes. Change if necessary.
every state/.style={thick, fill=gray!10}, % sets the properties for each ’state’ node
initial text=$ $, % sets the text that appears on the start arrow
}
\\begin{document}
\\begin{figure}[ht]
\\centering
\\begin{tikzpicture}
"""

_latex_document_end = """
\\end{tikzpicture}
%\\caption{Caption of the FSM}
%\\label{fig:my_label}
\\end{figure}
\\end{document}
"""

with open(f"{file_name}.tex", 'w', encoding="utf-8") as fp:
    fp.write(_latex_document_start)
    fp.write(_latex_nodes)
    fp.write(_latex_transitions)
    fp.write(_latex_document_end)

import os
import shutil

print(os.system(
    f'DEL /F C:\\Users\\Lukas\\Dropbox\\Informatik\\so2021\\"Formale Sprachen und Komplexität"\\04\\{file_name}.pdf'))
home_dir = os.system(f"pdflatex {file_name}.tex")
shutil.move(f"{file_name}.pdf", path)
os.system(f"DEL /F {file_name}.*")
print(len(new_states))
