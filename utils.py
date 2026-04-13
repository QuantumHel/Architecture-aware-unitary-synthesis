from collections import deque
import numpy as np
from scipy.linalg import cossin
from scipy.linalg import polar
from numpy.linalg import eigh, pinv
from scipy.linalg import sqrtm
import sys
import math
import multiprocessing
from multiprocessing import shared_memory
from a_star import BasicAStar

import os
os.environ['OPENBLAS_NUM_THREADS'] = '6'
os.environ['NUMEXPR_NUM_THREADS'] = '6'
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

def normalize(data):
    sq = 0
    for el in data:
        sq += np.square(el)

    sq = np.sqrt(sq)
    arr = data / sq
    return arr

def grey_code(dist, half):
    upper_bound = (2 ** (dist - 1) + 1) if half else 2 ** dist + 1

    for i in range(1, upper_bound):
        highest_index_diff = 0
        for j in range(dist):
            if (i >> j) & 1 != ((i - 1) >> j) & 1: highest_index_diff = j + 1
        grey_gate_queue.append("RZ")
        grey_gate_queue.append((dist - highest_index_diff, dist))

        grey_state_queue.append(grey_state[dist])
        grey_state[dist] ^= grey_state[dist - highest_index_diff]

def get_grey_gates(dist, half = False, all_gates = True, state_queue = False):
    global grey_gate_queue
    global grey_state
    global grey_state_queue

    grey_gate_queue = deque()
    grey_state_queue = deque()
    grey_state = {q: 1 << q for q in range(dist + 1)}

    grey_code(dist, half)

    long_range_gates = []
    for gate in grey_gate_queue:
        if gate == "RZ": continue
        if gate[1] - gate[0] > 1 and not all_gates: long_range_gates.append(gate)
        elif all_gates: long_range_gates.append(gate)

    if state_queue:
        return long_range_gates, grey_state_queue
    else:
        return long_range_gates



def generate_U(num_qubits):
    arr = normalize(np.array([math.cos(2 * np.pi * x - np.pi) + 1j*math.sin(2 * np.pi * x - np.pi) for x in np.random.random_sample((2 ** num_qubits) ** 2)]))
    # arr = normalize(np.array([x for x in np.random.random_sample((2 ** num_qubits) ** 2)]))
    mat_A = arr.reshape((2 ** num_qubits, 2 ** num_qubits))
    U, _, _ = np.linalg.svd(mat_A)
    return U

def simultaneous_diagonalization(s_x, s_y, tol=1e-12):
    eigvals, v = eigh(s_x)
    # sort ascending for determinism
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    v = v[:, idx]

    # group degenerate eigenvalues
    groups = []
    current = [0]
    for i in range(1, len(eigvals)):
        if abs(eigvals[i] - eigvals[i-1]) < tol:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)

    v_final = v.copy()
    diag_s_y = np.zeros_like(eigvals, dtype=float)

    for group in groups:
        if len(group) == 1:
            k = group[0]
            diag_s_y[k] = np.real_if_close((v[:, k].conj().T @ s_y @ v[:, k]))
            continue
        # refine basis inside degenerate block
        basis = v[:, group]                # n x m
        s_y_sub = basis.conj().T @ s_y @ basis  # m x m Hermitian
        evals_sub, U_sub = eigh(s_y_sub)
        v_final[:, group] = basis @ U_sub
        diag_s_y[group] = np.real_if_close(evals_sub)


    diag_s_x = np.real_if_close(eigvals)
    # clip small negative eigenvalues due to rounding
    diag_s_x = np.clip(diag_s_x, 0.0, None)
    diag_s_y = np.real_if_close(diag_s_y)
    return v_final, diag_s_x, diag_s_y

# def orthogonal_congruence_diagonalize(S, tol_eig=1e-1): #Lower value gives better succes chance for larger qubit count???? Look into this if time
#     """
#     Given a complex symmetric 4x4 matrix S = S.T,
#     returns real orthogonal A and diagonal D (complex phases)
#     such that D = A.T @ S @ A (up to numerical tolerance).

#     Handles degenerate eigenvalues in Re(S) by block-diagonalization
#     of Im(S) inside degenerate subspaces.
#     """
#     assert S.shape == (4,4)
#     assert np.allclose(S, S.T, atol=1e-10), "S must be symmetric (S==S.T)"

#     # Real/imag parts (real symmetric)
#     R = np.real(S)
#     ImS = np.imag(S)

#     # 1) Eigendecompose real symmetric R: use eigh to get real orthonormal Q
#     eigvals, Q = np.linalg.eigh(R)   # Q columns are eigenvectors (real)

#     # 2) Group eigenvalues into degenerate clusters
#     clusters = []
#     used = np.zeros(len(eigvals), dtype=bool)
#     for i in range(len(eigvals)):
#         if used[i]:
#             continue
#         # find indices j where eigvals[j] ~= eigvals[i]
#         idx = [j for j in range(len(eigvals)) if abs(eigvals[j] - eigvals[i]) < tol_eig]
#         for j in idx:
#             used[j] = True
#         clusters.append(idx)

#     # 3) For each cluster of size > 1, diagonalize ImS restricted to that subspace
#     Q_final = Q.copy()
#     for cluster in clusters:
#         if len(cluster) == 1:
#             continue
#         # form basis vectors for this cluster
#         cols = cluster
#         subQ = Q[:, cols]   # shape (4, k)
#         # Project ImS into this subspace: B = subQ.T @ ImS @ subQ  (real symmetric)
#         B = subQ.T @ ImS @ subQ
#         # B is real symmetric; diagonalize it to get an orthonormal transform U_block
#         bvals, Ublock = np.linalg.eigh(B)
#         # Replace the columns subQ @ Ublock into Q_final
#         Q_final[:, cols] = subQ @ Ublock

#     # Q_final should be real orthogonal
#     # enforce orthonormality numerically (e.g. via QR) if needed
#     # small re-orthonormalization:
#     U, _, Vh = np.linalg.svd(Q_final)
#     Q_final = U @ Vh   # now orthonormal and det = +/-
#     # enforce det=+1 by flipping sign of first column if needed
#     if np.linalg.det(Q_final) < 0:
#         Q_final[:, 0] = -Q_final[:, 0]

#     A = Q_final   # real orthogonal
#     return A

def orthogonal_congruence_diagonalize(S):
    """
    Given a complex symmetric matrix S = S.T,
    returns real orthogonal A (det +1) such that A.T @ S @ A
    is approximately diagonal.

    Uses a two-phase approach for numerical stability:

    Phase 1 – linear-combination eigenvectors:
        If Re(S) and Im(S) are simultaneously diagonalisable (as they must
        be for the matrices arising in KAK-type decompositions), the
        eigenvectors of Re(S) + α·Im(S) diagonalise both for almost any α.
        Sweeping several α values avoids the fragile eigenvalue-clustering
        heuristic and breaks accidental degeneracies.

    Phase 2 – Jacobi refinement:
        If any off-diagonal residual remains (due to numerical noise or an
        unlucky choice of α), iterative Givens rotations minimise the
        combined off-diagonal norm of both real and imaginary parts.
    """
    n = S.shape[0]
    assert np.allclose(S, S.T, atol=1e-10), "S must be symmetric (S == S.T)"

    R = np.real(S)
    J = np.imag(S)
    off_mask = ~np.eye(n, dtype=bool)

    # ── Phase 1: best initial basis from linear combinations ──
    best_Q = None
    best_err = np.inf

    alphas = [0.0, 1.0, -1.0, np.sqrt(2), -np.sqrt(2),
              np.pi, -np.pi, 0.5, -0.5, np.e, -np.e,
              np.sqrt(3), -np.sqrt(3), 0.1, -0.1, 2.0, -2.0,
              1.7, -1.7, 0.3, -0.3]

    for alpha in alphas:
        M = R + alpha * J
        _, Q = np.linalg.eigh(M)          # real orthonormal columns
        D = Q.T @ S @ Q
        err = np.max(np.abs(D[off_mask]))
        if err < best_err:
            best_err = err
            best_Q = Q.copy()
        if err < 1e-12:
            break

    Q = best_Q

    # ── Phase 2: Jacobi refinement sweeps ──
    # if best_err > 1e-12:
    #     print("Entered Jacobi refinement")
    #     D = Q.T @ S @ Q
    #     for _ in range(200):
    #         max_off = 0.0
    #         for i in range(n):
    #             for j in range(i + 1, n):
    #                 rij = np.real(D[i, j])
    #                 jij = np.imag(D[i, j])
    #                 mag = abs(rij) + abs(jij)
    #                 if mag < 1e-15:
    #                     continue
    #                 if mag > max_off:
    #                     max_off = mag

    #                 # Givens angle minimising
    #                 #   |cos2θ·rij − sin2θ·rd|² + |cos2θ·jij − sin2θ·jd|²
    #                 rd = (np.real(D[i, i]) - np.real(D[j, j])) / 2.0
    #                 jd = (np.imag(D[i, i]) - np.imag(D[j, j])) / 2.0

    #                 p = rij * rij + jij * jij - rd * rd - jd * jd
    #                 q = -2.0 * (rij * rd + jij * jd)

    #                 if abs(p) < 1e-30 and abs(q) < 1e-30:
    #                     continue

    #                 # Two candidate angles (minima vs maxima of the cost)
    #                 base = np.arctan2(q, p) / 4.0
    #                 best_theta = base
    #                 best_cost = np.inf
    #                 for th in (base, base + np.pi / 4.0):
    #                     c2 = np.cos(2.0 * th)
    #                     s2 = np.sin(2.0 * th)
    #                     cost = (c2 * rij - s2 * rd) ** 2 + (c2 * jij - s2 * jd) ** 2
    #                     if cost < best_cost:
    #                         best_cost = cost
    #                         best_theta = th

    #                 c, s = np.cos(best_theta), np.sin(best_theta)
    #                 # Apply Givens: D ← G.T D G,  Q ← Q G
    #                 for k in range(n):
    #                     di, dj = D[k, i], D[k, j]
    #                     D[k, i] = c * di - s * dj
    #                     D[k, j] = s * di + c * dj
    #                 for k in range(n):
    #                     di, dj = D[i, k], D[j, k]
    #                     D[i, k] = c * di - s * dj
    #                     D[j, k] = s * di + c * dj
    #                 for k in range(n):
    #                     qi, qj = Q[k, i], Q[k, j]
    #                     Q[k, i] = c * qi - s * qj
    #                     Q[k, j] = s * qi + c * qj

    #         if max_off < 1e-13:
    #             break

    # Enforce det = +1
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    return Q

# Source - https://stackoverflow.com/a/104436
# Posted by Eli Bendersky, modified by community. See post 'Timeline' for change history
# Retrieved 2026-04-13, License - CC BY-SA 4.0
def permutations(elements):
    if len(elements) <= 1:
        yield elements
        return
    for perm in permutations(elements[1:]):
        for i in range(len(elements)):
            # nb elements[0:1] works in both string and list contexts
            yield perm[:i] + elements[0:1] + perm[i:]


def permute_and_negate(A, B, u, v):
    if np.allclose(A.T @ u @ A, B.T @ v @ B):
      print("No permutation needed")
      return A

    indices = [0, 1, 2, 3]
    perms = permutations(indices)
    for perm in perms:
        idx = np.empty_like(perm)
        idx[perm] = np.arange(len(perm))
        A_perm = A[:, idx]
        for i in range(4):
            A_perm_copy = A_perm.copy()
            A_perm_copy[:, i] *= -1
            if np.allclose(A_perm_copy.T @ u @ A_perm_copy, B.T @ v @ B):
                return A_perm_copy
    # print(A.T @ u @ A)
    # print("///////////")
    # print(B.T @ v @ B)
    # raise ValueError(f"No valid permutation found")
    return A

def compute_csd(U, tol=1e-12):
    n = U.shape[0] // 2

    X = U[:n, :n]
    Y = U[:n, n:]
    W = U[n:, :n]
    Z = U[n:, n:]

    # Hermitian PSD factors (more stable than polar for our decomposition)
    XX = X @ X.conj().T
    YY = Y @ Y.conj().T

    s_x = sqrtm(XX)
    s_y = sqrtm(YY)
    # enforce Hermitian (numerical)
    s_x = 0.5 * (s_x + s_x.conj().T)
    s_y = 0.5 * (s_y + s_y.conj().T)

    # unitary factors: u_x, u_y such that X = s_x @ u_x, Y = s_y @ u_y
    # Use pseudo-inverse for s_x,s_y in case of singular values (stable)
    s_x_pinv = pinv(s_x)
    s_y_pinv = pinv(s_y)
    u_x = s_x_pinv @ X
    u_y = s_y_pinv @ Y

    v, diag_s_x, diag_s_y = simultaneous_diagonalization(s_x, s_y, tol=tol)
    sigma_x = np.diag(diag_s_x)
    delta_y = np.diag(diag_s_y)

    M0 = Z @ u_y.conj().T @ s_x - W @ u_x.conj().T @ s_y
    M = M0 @ v

    u = np.block([[v, np.zeros((n, n), dtype=U.dtype)],
                [np.zeros((n, n), dtype=U.dtype), M]])

    cs = np.block([[sigma_x, delta_y],
                   [-delta_y, sigma_x]])

    vh = np.block([[v.conj().T @ u_x, np.zeros((n, n), dtype=U.dtype)],
                    [np.zeros((n, n), dtype=U.dtype), v.conj().T @ u_y]])


    return u, cs, vh

def is_unitary(U):
    UU = clean_matrix(U @ np.conj(U.T))
    return np.allclose(UU, np.eye(len(UU)))

def generate_random_rz_multiplexer_unitary(num_qubits):
    U = generate_U(num_qubits)

    sub_mat_dim = num_qubits - 1
    Z = np.zeros((2 ** sub_mat_dim, 2 ** sub_mat_dim))

    u, _, _ = compute_csd(U)

    u_1 = u[:2 ** sub_mat_dim, :2 ** sub_mat_dim]
    u_2 = u[2 ** sub_mat_dim:, 2 ** sub_mat_dim:]

    u_1_u_2_dgr = u_1 @ np.conj(u_2.T)

    eigval, _ = np.linalg.eig(u_1_u_2_dgr)

    if -1 or 1 in eigval: diag = np.diag(eigval)
    else: diag = np.diag([np.sqrt(x) for x in eigval])

    block_diag = np.block([[diag, Z],
                            [Z, np.conj(diag.T)]])
    return block_diag

def generate_random_rz_multiplexer_unitary_fast(num_qubits):
    num_rzs = 2 ** (num_qubits - 1)
    rzs =  []
    phis = []
    for _ in range(num_rzs):
        phi = np.random.random() * 2 * np.pi - np.pi
        phis.append(phi)
        rzs.append(math.cos(phi) - 1j*math.sin(phi))

    for phi in phis:
        rzs.append(math.cos(phi) + 1j*math.sin(phi))

    return np.diag(rzs)

def extract_single_qubit_unitaries(mat):
    half = len(mat) // 2
    for i in range(half):
        j = int((f"{{:0>{int(math.log2(half))}b}}".format(i))[::-1], 2) #Don't ask
        first = mat[j][j]
        second = mat[half + j][half + j]
        yield np.diag([first, second])

def extract_angles(unitaries):
    for unitary in unitaries:
        value = unitary[0][0]

        if value > 1: value = 1
        elif value < -1: value = -1

        ang = math.acos(np.real(value))

        if np.round(math.sin(ang), 32) == np.round(np.imag(value), 32): ang = -ang
        yield ang

def angles_from_diag(diag):
    angles = []
    half = len(diag) // 2
    for i in range(half):
        j = int((f"{{:0>{int(math.log2(half))}b}}".format(i))[::-1], 2)
        angles.append(np.angle(diag[half + i][half + i]))
    return angles

def extract_angles_from_eigvals(eigvals):
    for value in eigvals:
        ang = math.acos(np.real(value))

        if np.round(math.sin(ang), 10) == np.round(np.imag(value), 10): ang = -ang
        yield ang

def random_angles(n):
    return [np.random.random() * 2 * np.pi - np.pi for _ in range(n)]

def clean_matrix(M):
    M = M.copy()
    for i in range(len(M)):
        for j in range(len(M)):
            if np.abs(M[i][j]) < 1e-12: M[i][j] = float(0.0)
            if np.abs(np.imag(M[i][j])) < 1e-12: M[i][j] = np.real(M[i][j])
            M[i][j] = '{0:.12}'.format(M[i][j])
    return M

def _möttönen_transformation(start, stop, n, num_controls, global_angles, transformed_angles_name, gray_code):
    existing_transformed_angles = shared_memory.SharedMemory(name=transformed_angles_name)
    transformed_angles = np.ndarray((n,), buffer=existing_transformed_angles.buf)

    power = math.pow(2, -num_controls)
    for i in range(start, stop):
        temp = 0
        g_m = i ^ (i >> 1) if gray_code == None else gray_code[i]
        for j in range(n):
            dot_product = bin(g_m & j).count('1') % 2
            temp +=  -global_angles[j]  if dot_product == 1 else global_angles[j]
        transformed_angles[i] = power * temp * 2

    existing_transformed_angles.close()

def möttönen_transformation(multiplexer_angles, gray_code = None):
        global_angles = np.array(multiplexer_angles)
        n = len(multiplexer_angles)
        transformed_angles = np.zeros(n)
        num_controls = int(math.log2(n))

        if num_controls >= 12:
            shm_transformed_angles = shared_memory.SharedMemory(create=True, size=transformed_angles.nbytes)
            new_shm_transformed_angles = np.ndarray(transformed_angles.shape, dtype=transformed_angles.dtype, buffer=shm_transformed_angles.buf)
            new_shm_transformed_angles[:] = transformed_angles[:]

            jobs = []
            cores = multiprocessing.cpu_count()

            for i in range(cores):
                start = int((n / cores) * i)
                stop = min(int((n / cores) * (i + 1)), n)
                p = multiprocessing.Process(target = _möttönen_transformation, args=(start, stop, n, num_controls, global_angles, shm_transformed_angles.name, gray_code))
                jobs.append(p)
                p.start()

            for proc in jobs:
                proc.join()

            transformed_angles = new_shm_transformed_angles.copy()
            shm_transformed_angles.close()
            shm_transformed_angles.unlink()
        else:
            power = math.pow(2, -num_controls)
            for i in range(n):
                temp = 0
                g_m = i ^ (i >> 1) if gray_code == None else gray_code[i]
                for j in range(n):
                    dot_product = bin(g_m & j).count('1') % 2
                    temp +=  -global_angles[j]  if dot_product == 1 else global_angles[j]
                transformed_angles[i] = power * temp * 2
        return transformed_angles


def get_zyz_angles(U):
    """
    Decomposes a 2x2 unitary matrix U into ZYZ Euler angles.
    Returns (phi, theta, lam) such that:
    U = Rz(phi) @ Ry(theta) @ Rz(lam)
    """
    # Extract elements
    u00 = U[0, 0]
    u10 = U[1, 0]
    u11 = U[1, 1]

    # 1. Calculate Theta
    # Use abs() to handle complex magnitudes
    # 2 * atan2(sin_part, cos_part)
    theta = 2 * np.arctan2(np.abs(u10), np.abs(u00))

    # Define a small tolerance for float comparison
    TOL = 1e-64

    # 2. Calculate Phi and Lambda (Handling singularities)
    if np.abs(u10) < TOL: # Theta is approx 0
        # u10 is 0, so arg(u10) is undefined.
        # We only know phi + lam = 2 * arg(u11)
        lam = 0.0
        phi = 2 * np.angle(u11)

    elif np.abs(u00) < TOL: # Theta is approx Pi
        # u00 (and u11) is 0, so arg(u11) is undefined.
        # We only know phi - lam = 2 * arg(u10)
        lam = 0.0
        phi = 2 * np.angle(u10)

    else: # General case
        # angle(u11) = (phi + lam)/2
        # angle(u10) = (phi - lam)/2

        sum_phases = 2 * np.angle(u11)
        diff_phases = 2 * np.angle(u10)

        phi = (sum_phases + diff_phases) / 2
        lam = (sum_phases - diff_phases) / 2

    return phi, theta, lam

def check_equivalence_up_to_phase(u_orig, u_recon):
    # 1. Compute the overlap (inner product)
    # If u_orig == alpha * u_recon, then trace(u_orig^dag @ u_recon) = trace(conj(alpha) * I * 4)
    overlap = np.trace(u_orig.conj().T @ u_recon)

    # 2. The magnitude of the overlap should be equal to the dimension (4)
    dim = u_orig.shape[0]
    # print(dim)
    # print(overlap)
    # print(np.abs(overlap))
    if not np.isclose(np.abs(overlap), dim, atol=1e-5):
        print(f"FAILED: Matrices are not equivalent. Overlap magnitude: {np.abs(overlap)}")
        return False, None

    # 3. The global phase is the "angle" of the overlap
    # We normalize by the dimension to isolate alpha
    phase_factor = overlap / dim

    print(f"SUCCESS: Matrices are equivalent.")
    print(f"Global Phase Difference: {phase_factor:.5f}")

    return True, phase_factor

def get_subset_of_neighbors(neighbors, subset_nodes):
        subset = neighbors.copy()
        for key in neighbors.copy().keys():
            if key not in subset_nodes: subset.pop(key)
            else: subset[key] = subset[key].intersection(subset_nodes)
        return subset

def get_path(neighbors, subset_nodes, source, target):
    new_neighbors = get_subset_of_neighbors(neighbors, subset_nodes)
    AStar = BasicAStar(new_neighbors)
    return AStar.astar(source, target)

def project_to_SU(U, n):
    detU = np.linalg.det(U)
    assert detU != 0, "Matrix is not unitary!"
    phase = detU ** (1 / n)
    return U / phase, phase

def rz(angle):
    return np.diag([np.exp(-1j * angle / 2), np.exp(1j * angle / 2)])

def rx(angle):
    return np.array([
        [np.cos(angle / 2), -1j * np.sin(angle / 2)],
        [-1j * np.sin(angle / 2), np.cos(angle / 2)]
    ])

def ry(angle):
    return np.array([
        [np.cos(angle / 2), -np.sin(angle / 2)],
        [np.sin(angle / 2), np.cos(angle / 2)]
    ])