from collections import deque
from functools import reduce
import math
from utils import get_path
from cluster_finding import find_closest_cluster
import numpy as np
from gray_synth import synth_cnot_phase_aam

class RoutedMultiplexer(object):
    def __init__(self, multiplexer_angles = None, coupling_map = None, num_qubits = 5, reverse = True):
        assert num_qubits >= 2

        self.multiplexer_angles = multiplexer_angles
        self.coupling_map = coupling_map
        self.reverse = reverse

        self.num_qubits = num_qubits if multiplexer_angles == None else int(math.log2(len(multiplexer_angles)) + 1)
        self.num_controls = self.num_qubits - 1

        if self.multiplexer_angles == None: 
            self.multiplexer_angles = [0.123 for x in range(2 ** self.num_controls)]

        self.multiplexer_angles = np.array(self.multiplexer_angles)
        if self.coupling_map == None: self.coupling_map = [[x, y] for x in range(self.num_qubits) for y in range(self.num_qubits) if x != y]

        self.neighbors = self.get_neighbors()
        self.vertices = list(self.neighbors.copy().keys())
        self.found_all_terms = False

        assert len(self.vertices) >= self.num_qubits, "Not enough qubits on the hardware."
    
    def __str__(self):
        has_grey_to_arch = hasattr(self, "grey_to_arch_map")
        has_arch_to_grey = hasattr(self, "arch_to_grey_map")
        has_root = hasattr(self, "root")
        has_optimal_neighborhood = hasattr(self, "optimal_neighborhood")
        return f"Num_qubits: {self.num_qubits}, Root: {self.root if has_root else None}, Grey_to_arch: {self.grey_to_arch_map if has_grey_to_arch else None}, Arch_to_grey: {self.arch_to_grey_map if has_arch_to_grey else None}, Optimal_neighborhood: {self.optimal_neighborhood if has_optimal_neighborhood else None}"

    def get_neighbors(self):
        """
        Produces the set of neighbors for each physical qubits
        """
        neighbors = {}
        for edge in self.coupling_map:
            if neighbors.get(edge[0]) == None: neighbors[edge[0]] = set()
            if neighbors.get(edge[1]) == None: neighbors[edge[1]] = set()

            neighbors[edge[0]].add(edge[1])
            neighbors[edge[1]].add(edge[0])

        return neighbors
    
    
    def find_optimal_neighborhood_closest_cluster(self):
        """
        This function does the initial qubit mapping, and produces the initial "optimal neighborhood", which is used when decomposing the UC gate.
        """
        cluster_e, dists = find_closest_cluster(self.num_qubits, self.neighbors, method="auto")
        self.pairwise_dists = dists
        root = cluster_e[0]
        paths = {root: [root]}
        for node in cluster_e:
            if node != root:
                path = list(get_path(self.neighbors, set(cluster_e), root, node))
                paths[node] = path
                
        grey_to_arch = {i: n for i, n in enumerate(cluster_e)}

        return (paths, grey_to_arch)
                    

    def recompute_optimal_neighborhood(self):
        """
        Recomputes the "optimal neighborhood" base on the current "arch_qubits"
        """
        new_optimal_neighborhood = {}

        for node in self.optimal_neighborhood.keys():
            path = list(get_path(self.neighbors, self.arch_qubits, self.root, node))
            new_optimal_neighborhood[node] = path
        self.optimal_neighborhood = new_optimal_neighborhood

    def map_grey_qubits_to_arch_unitary_synth(self):
        """
        Calculates the "grey_to_arch" map, which is the inverse of the initial qubit map. Also initializes the "root", or the target qubit for the
        current recursion level, and the furthest physical qubit from the root.
        """
        self.optimal_neighborhood, self.grey_to_arch_map = self.find_optimal_neighborhood_closest_cluster()
        self.arch_to_grey_map = {}
        for key, value in self.grey_to_arch_map.items():
            self.arch_to_grey_map[value] = key
        
        self.arch_qubits = list(self.grey_to_arch_map.copy().values())
        self.root = self.grey_to_arch_map[self.num_controls]
        self.furthest_node = self.grey_to_arch_map[0]

    def long_range_cnot_cost(self, dist):
        """
        Equation (11) from the paper.
        """
        return 4 * dist - 4 if dist > 1 else 1

    def get_optimal_gery_code(self):
        """
        Calculates the optimal Gray code and cost estimate based on the current qubit layout.
        """
        dists = {} # Distances from the root to other physical qubits in the layout
        for node in self.arch_to_grey_map.keys():
            if node == self.root: continue
            dists[node] = self.pairwise_dists[self.root][node]

        dists_list = list(zip(dists.keys(), dists.values()))
        dists_list.sort(key=lambda tup: tup[1])
        
        arch_cnots = {}
        for index, node in enumerate(dists_list):
            arch_cnots[index] = node

        
        upper_bound = 2 ** (self.num_controls) + 1
        self.grey_gate_queue = deque()
        self.grey_state_queue = deque()


        grey_state = {q: 1 << q for q in range(self.num_qubits)}
        for i in range(1, upper_bound): # This loop calculates the Gray code
            highest_index_diff = 0
            for j in range(self.num_controls):
                if (i >> j) & 1 != ((i - 1) >> j) & 1: highest_index_diff = j + 1
            cnot = arch_cnots[highest_index_diff - 1]
            self.grey_gate_queue.append((self.arch_to_grey_map[cnot[0]], self.arch_to_grey_map[self.root]))

            self.grey_state_queue.append(grey_state[self.num_controls])
            grey_state[self.num_controls] ^= grey_state[self.arch_to_grey_map[cnot[0]]]

        self.grey_code = list(map(lambda x: x ^ (1 << self.num_controls), list(self.grey_state_queue.copy()))) # Store the Gray code


        cnots = {self.num_controls -1: 2} # The CNOT frequencies for the Gray code
        for i in range(self.num_qubits - 2):
            cnots[self.num_controls - 2 - i] = 2
            for j in range(i):
                cnots[ self.num_controls - 1 - j] *= 2

        # Cost estimate is calculates based on the CNOT frequencies and the distances.
        total_cost = reduce(lambda x, y: x + y[1] * self.long_range_cnot_cost(y[0]), zip(map(lambda z: z[1], dists_list), cnots.values()), 0)
        return total_cost

    def cancel_or_append(self, cnot, ignore):
        """
        This function implements heuristic from section III.C of the paper and updates the discovered Gray terms.
        """
        prev_gate = self.gate_queue.pop()
        cancelled = True
        if not ignore or prev_gate != cnot:
            self.gate_queue.append(prev_gate)
            self.gate_queue.append(cnot)
            cancelled = False
        
        if cnot[1] == self.num_controls:
            if self.state[self.num_controls] not in self.discovered_pp_terms:
                self.discovered_pp_terms.add(self.state[self.num_controls])
                self.gate_queue.append(("RZ", self.state_to_angle_dict[self.state[self.num_controls]], self.num_controls))
            elif ignore == True and self.state[self.num_controls] in self.discovered_pp_terms and not cancelled:
                self.gate_queue.pop()
                self.state[cnot[1]] ^= self.state[cnot[0]]

        
    
    def long_range_cnot(self, arch_path, ignore = False):
        """
        This is the long range CNOT ladder
        """

        grey_path = list(reversed(list(map(lambda arch_qubit: self.arch_to_grey_map[arch_qubit], arch_path))))
        grey_path_dist = len(grey_path)
        for i in range(grey_path_dist - 1):
            self.state[grey_path[i + 1]] ^= self.state[grey_path[i]]
            self.cancel_or_append((grey_path[i], grey_path[i + 1]), ignore)

        for j in range(grey_path_dist - 2, 0, -1):
            self.state[grey_path[j]] ^= self.state[grey_path[j - 1]]
            self.cancel_or_append((grey_path[j - 1], grey_path[j]), ignore)

        for k in range(1, grey_path_dist - 1):
            self.state[grey_path[k + 1]] ^= self.state[grey_path[k]]
            self.cancel_or_append((grey_path[k], grey_path[k + 1]), ignore)

        for l in range(grey_path_dist - 2, 1, -1):
            self.state[grey_path[l]] ^= self.state[grey_path[l - 1]]
            self.cancel_or_append((grey_path[l - 1], grey_path[l]), ignore)

    def reset_state(self):
        for i in range(self.num_qubits):
            if (self.state[self.num_controls] >> i) & 1 == 1: 
                arch_qubit = self.grey_to_arch_map[i]
                arch_path = self.optimal_neighborhood[arch_qubit]
                self.long_range_cnot(arch_path, False)

    def find_missing_terms(self, unfound_terms):
        """
        This is the Gray synth fallback
        """
        parity_matrix = []
        for term in unfound_terms:
            temp = []
            for i in range(self.num_qubits):
                if (term >> i) & 1 == 1: temp.append(1)
                else: temp.append(0)

            parity_matrix.append(temp)

        parity_matrix = list(np.array(parity_matrix).transpose())
        for i in range(len(parity_matrix)):
            parity_matrix[i] = list(parity_matrix[i])

        angles = [0.123 for x in range(len(parity_matrix[0]))]
        qc = synth_cnot_phase_aam(parity_matrix, angles)
        synth_cnots = deque()
        for instruction in qc.data:
            indices = [qc.find_bit(q).index for q in instruction.qubits]
            if len(indices) > 1: synth_cnots.append((indices[0], indices[1]))

        # Indices wrong way around?

        for gate in synth_cnots:
            ctrl_qubit = gate[0]
            arch_qubit = self.grey_to_arch_map[ctrl_qubit]
            arch_path = self.optimal_neighborhood[arch_qubit]

            self.long_range_cnot(arch_path, False)

    def execute_gates(self):
        """
        Applies the CNOT and RZ gates to decompose the UC gate based to the optimal Gray code the physical qubit layout
        """
        self.pp_terms = set([x for x in range(2 ** self.num_controls, 2 ** self.num_qubits)])
        
        self.discovered_pp_terms = set([2 ** self.num_controls])
        self.state = {q: 1 << q for q in range(self.num_qubits)}
        self.state_to_angle_dict = dict(zip(self.grey_state_queue, self.multiplexer_angles))
        init_state = self.state.copy()

        self.gate_queue = deque()
        self.gate_queue.append(("RZ", self.multiplexer_angles[0], self.num_controls))

        for gate in self.grey_gate_queue:
            ctrl_qubit = gate[0]
            arch_qubit = self.grey_to_arch_map[ctrl_qubit]
            arch_path = self.optimal_neighborhood[arch_qubit]
            dist = len(arch_path) - 1
            ignore = False if dist > 1 else True
            self.long_range_cnot(arch_path, ignore)


        if init_state != self.state:
            self.reset_state()
        unfound_terms = self.pp_terms - self.discovered_pp_terms

        gray_synth_fallback = False
        if len(unfound_terms) > 0:
            self.reset_state()
            self.find_missing_terms(unfound_terms)
            gray_synth_fallback = True

        circuit_length = len([gate for gate in self.gate_queue if gate[0] != "RZ"])

        if len(self.discovered_pp_terms) != len(self.pp_terms):
            print(f"Found {len(self.discovered_pp_terms)}/{len(self.pp_terms)} phase polynomial terms.")

        if self.state != init_state:
            print("State was not reset correctly!")

        print(list(self.gate_queue)[-100:])
        exit()
        self.cx_count = circuit_length
        return gray_synth_fallback, self.gate_queue.copy()

    
    def replace_mapped_angles(self, new_angles, reverse = True):
        """
        We use this instead of the execute_gates in order to be more efficient, since we dont need to compute everything again,
        just apply different RZ gates with the same layout
        """
        assert self.gate_queue != None
        old_gates = self.gate_queue.copy()
        new_gates = deque()
        
        while True:
            try:
                gate = old_gates.pop() if reverse else old_gates.popleft()
                if gate[0] != "RZ": new_gates.append(gate)
                else: 
                    new_gates.append(("RZ", new_angles[np.where(self.multiplexer_angles == gate[1])[0][0]], self.num_controls))
            except Exception as e:
                break
        return new_gates
    

    def copy(self):
        cp = RoutedMultiplexer(list(self.multiplexer_angles.copy()), self.coupling_map.copy(), self.num_qubits, self.reverse)
        cp.num_qubits = self.num_qubits
        cp.num_controls = self.num_controls
        cp.neighbors = self.neighbors.copy()
        cp.vertices = self.vertices.copy()
        cp.root = self.root
        cp.furthest_node = self.furthest_node
        cp.arch_to_grey_map = self.arch_to_grey_map.copy()
        cp.grey_to_arch_map = self.grey_to_arch_map.copy()
        cp.arch_qubits = self.arch_qubits.copy()
        cp.optimal_neighborhood = self.optimal_neighborhood.copy()
        cp.pairwise_dists = self.pairwise_dists
        cp.grey_code = self.grey_code
        cp.grey_gate_queue = self.grey_gate_queue
        cp.grey_state_queue = self.grey_state_queue
        return cp