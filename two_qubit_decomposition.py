import numpy as np
from utils import orthogonal_congruence_diagonalize, get_zyz_angles, ry, rz, rx
from scipy.optimize import linear_sum_assignment
from collections import deque
from qiskit.compiler import transpile
from qiskit_aer import Aer
import sys

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

sigma_y = np.array([[0, -1j],
                    [1j, 0]])


xi = np.exp(1j * np.pi / 4)

cnot_1_2 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]])

cnot_2_1 = np.array([[1, 0, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0]])

cnot_1_2 = cnot_1_2 * xi
cnot_2_1 = cnot_2_1 * xi


E = np.array([[1 / np.sqrt(2), 1j / np.sqrt(2), 0, 0],
              [0, 0, 1j / np.sqrt(2), 1 / np.sqrt(2)],
              [0, 0, 1j / np.sqrt(2), -1 / np.sqrt(2)],
              [1 / np.sqrt(2), -1j / np.sqrt(2), 0, 0]])

E_dgr = np.conjugate(E).T

I = np.eye(2)

sigma_y_kron_2 = np.kron(sigma_y, sigma_y)

def print_circ_unitary(qc):
    qc = qc.copy()
    qc.save_unitary()
    simulator = Aer.get_backend('aer_simulator')
    qc = transpile(qc, simulator)

    result = simulator.run(qc).result()
    unitary = result.get_unitary(qc)
    # phase = np.linalg.det(U) ** (1 / 4)

    print("Circuit unitary:\n", np.asarray(unitary).round(5))

def project_to_SU4(U):
    """Project a 4x4 unitary onto SU(4) by dividing out the global phase.
 
    Chooses the fourth-root branch whose phase angle is closest to zero,
    which keeps the result continuous and avoids the +/-pi branch-cut
    discontinuity of Python's principal root.
    """
    detU = np.linalg.det(U)
    if abs(detU) < 1e-15:
        detU = 1e-15
    det_angle = np.angle(detU)
    det_mag = abs(detU)
    mag4 = det_mag ** 0.25
 
    # Four candidate phase angles: (det_angle + 2*pi*k) / 4  for k = 0..3
    # Pick the one closest to 0 (wrapping into (-pi, pi]).
    best_angle = None
    best_dist = np.inf
    for k in range(4):
        a = (det_angle + 2.0 * np.pi * k) / 4.0
        # Wrap to (-pi, pi]
        a = (a + np.pi) % (2.0 * np.pi) - np.pi
        if abs(a) < best_dist:
            best_dist = abs(a)
            best_angle = a
 
    phase = mag4 * np.exp(1j * best_angle)
    return U / phase, phase

def project_to_SU2(U):
    detU = np.linalg.det(U)
    # This might fail
    # assert detU != 0
    if detU == 0: detU = 1e-15
    return U / detU ** (1 / 2)

def gamma_map(u):
    assert len(u) == 4
    return u @ sigma_y_kron_2 @ u.T @ sigma_y_kron_2

def _robust_angle_sort(eigvals):
    """Sort eigenvalue angles on the unit circle, avoiding +/-pi discontinuity.
 
    Finds the largest angular gap between adjacent eigenvalues on the circle,
    places the branch cut inside that gap, and then sorts.  This guarantees
    that eigenvalues which are close on the circle stay adjacent after sorting,
    even when they straddle the conventional +/-pi cut.
 
    Returns the angles in ascending order.
    """
    angles = np.angle(eigvals)
    n = len(angles)
    if n == 0:
        return angles
 
    # Sort once to find gaps
    idx = np.argsort(angles)
    sa = angles[idx]
 
    # Circular gaps (including the wrap-around)
    gaps = np.empty(n)
    for i in range(n - 1):
        gaps[i] = sa[i + 1] - sa[i]
    gaps[n - 1] = sa[0] + 2.0 * np.pi - sa[n - 1]
 
    # Place the cut in the middle of the largest gap
    gi = np.argmax(gaps)
    if gi < n - 1:
        cut = sa[gi] + gaps[gi] / 2.0
    else:
        cut = sa[n - 1] + gaps[n - 1] / 2.0
 
    # Shift so the cut maps to +pi (the atan2 branch cut)
    shift = np.pi - cut
    shifted = np.angle(eigvals * np.exp(1j * shift))
    order = np.argsort(shifted)
    return angles[order]

def _pair_conjugate_angles(eigvals):
    """Extract two canonical angles from eigenvalues that form conjugate pairs.
 
    After the Delta correction in extract_diagonal, gamma_map(U*Delta) has
    eigenvalues {e^{ia}, e^{-ia}, e^{ib}, e^{-ib}}.  This function identifies
    the pairs using the product criterion (lam_i * lam_j ~ 1 for conjugates)
    and returns (a, b) with 0 <= a <= b, using the ratio trick to avoid the
    +/-pi branch-cut issue in np.angle.
    """
    n = len(eigvals)
    used = [False] * n
    pair_angles = []
 
    for i in range(n):
        if used[i]:
            continue
        best_j = -1
        best_d = np.inf
        for j in range(i + 1, n):
            if used[j]:
                continue
            # For a conjugate pair on the unit circle: lam_i * lam_j = 1
            d = abs(eigvals[i] * eigvals[j] - 1.0)
            if d < best_d:
                best_d = d
                best_j = j
        used[i] = True
        used[best_j] = True
 
        # For conjugate pair e^{ia} and e^{-ia}, Re(lambda) = cos(a).
        # arccos(cos(a)) = |a| for a in [-pi, pi], giving the pair angle
        # in [0, pi] without any branch-cut ambiguity.
        cos_a = (np.real(eigvals[i]) + np.real(eigvals[best_j])) / 2.0
        pair_angles.append(np.arccos(np.clip(cos_a, -1.0, 1.0)))
 
    pair_angles.sort()
    return tuple(pair_angles)       # (a, b) with a <= b

def extract_tensor_factors(M):
    M_reshaped = M.reshape(2, 2, 2, 2)
    M_reordered = M_reshaped.transpose(0, 2, 1, 3)
    M_flat = M_reordered.reshape(4, 4)
    
    u, s, vh = np.linalg.svd(M_flat)
    
    a = u[:, 0].reshape(2, 2)
    b = vh[0, :].reshape(2, 2)
    
    a = a * np.sqrt(s[0])
    b = b * np.sqrt(s[0])
    
    # Note: Setting a = a / np.sqrt(np.linalg.det(a)) directly may crash the process due to the determinant being zero.
    a = a / np.sqrt(np.linalg.det(a)) if np.linalg.det(a) != 0 else a / 1e-15
    b = b / np.sqrt(np.linalg.det(b)) if np.linalg.det(b) != 0 else b / 1e-15
    
    return a, b

def get_single_qubit_unitaries(U_E, k_E):
    S_U = U_E @ (U_E.T)
    S_k = k_E @ (k_E.T)

    A_U = orthogonal_congruence_diagonalize(S_U)
    B_k = orthogonal_congruence_diagonalize(S_k)

    D_U = np.diag(A_U.T @ S_U @ A_U)
    D_k = np.diag(B_k.T @ S_k @ B_k)

    # Find the column permutation that aligns D_k to D_U

    # Hungarian algorithm on angle-distance cost matrix
    ang_U = np.angle(D_U)
    ang_k = np.angle(D_k)
    # Circular distance: min(|a-b|, 2π-|a-b|)
    diff = np.abs(ang_U[:, None] - ang_k[None, :])
    cost = np.minimum(diff, 2.0 * np.pi - diff)
    _, perm = linear_sum_assignment(cost)
    B_k = B_k[:, perm]

    # Each column of an orthogonal eigenvector matrix has an arbitrary ±1 sign.
    # Try all 2^4 = 16 sign combinations for B_k and keep the one that makes
    # C_tilde closest to a rank-1 matrix (smallest s[1]/s[0]).
    best_B = B_k.copy()
    best_rank1 = np.inf
    for sign_bits in range(16):
        signs = np.array([1 - 2 * ((sign_bits >> i) & 1) for i in range(4)], dtype=float)
        B_cand = B_k * signs[None, :]

        # Quick check: det(A_U @ B_cand.T) must be positive for the
        # downstream math; skip if not.
        if np.real(np.linalg.det(A_U @ B_cand.T)) < 0:
            continue

        C_cand = np.conjugate(k_E).T @ B_cand @ A_U.T @ U_E
        C_tilde_cand = E @ C_cand @ E_dgr
        M_flat = C_tilde_cand.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
        s = np.linalg.svd(M_flat, compute_uv=False)
        ratio = s[1] / s[0] if s[0] > 1e-15 else 0.0
        if ratio < best_rank1:
            best_rank1 = ratio
            best_B = B_cand.copy()
    
        B_k = best_B

    if np.real(np.linalg.det(A_U @ B_k.T)) < 0:
            A_U[:, 0] = -A_U[:, 0]

    C = np.conjugate(k_E).T @ B_k @ A_U.T @ U_E

    A_tilde =  E @ A_U @ B_k.T @ E_dgr
    C_tilde = E @ C @ E_dgr

    a, b = extract_tensor_factors(A_tilde)
    c, d = extract_tensor_factors(C_tilde)

    return a, b, c, d

def _fix_global_phase(recon, U, a):
    """Correct the global phase of the reconstruction by adjusting a.
 
    The reconstruction and U should differ by at most a unit scalar.
    We absorb that scalar into the (a x b) factor by multiplying a
    by the conjugate of the phase mismatch.
    """
    phase_diff = np.trace(np.conjugate(recon).T @ U) / 4.0
    # phase_diff should be a unit complex number; absorb into a
    if abs(phase_diff) > 1e-12:
        correction = np.conj(phase_diff) / abs(phase_diff)
        a = a * correction
    return a

def extract_diagonal(u, source):
    # print(u)
    U, phase = project_to_SU4(u)
    M = gamma_map(U.T).T

    t_1 = M[0][0]
    t_2 = M[1][1]
    t_3 = M[2][2]
    t_4 = M[3][3]

    # psi = np.atan2(np.imag(t_1 + t_2 + t_3 + t_4), np.real(t_1 + t_4 - t_3 - t_2)) # Swap t_4 and t_2
    num = np.imag(t_1 + t_2 + t_3 + t_4)
    den = np.real(t_1 + t_4 - t_3 - t_2)
    if abs(num) < 1e-12 and abs(den) < 1e-12:
        psi = 0.0
    else:
        psi = np.atan2(num, den) # Swap t_4 and t_2

    Delta =  cnot_1_2 @ np.kron(I, rz(psi)) @ cnot_1_2
    gamma_U_Delta = gamma_map(U @ Delta)
    eigvals = np.linalg.eigvals(gamma_U_Delta)

    a_angle, b_angle = _pair_conjugate_angles(eigvals)
    theta = (a_angle - b_angle) / 2.0
    phi   = -(a_angle + b_angle) / 2.0

    U_E = (E_dgr @ U @ Delta @ E)

    kernel = cnot_1_2 @ np.kron(rx(theta + np.pi), rz(phi)) @ cnot_1_2 # Add + np.pi to get correct eigenvalues
    k_E = (E_dgr @ kernel @ E)

    a, b, c, d = get_single_qubit_unitaries(U_E, k_E)

    recon = np.kron(a, b) @ kernel @ np.kron(c, d) @ cnot_1_2 @ np.kron(I, rz(-psi)) @ cnot_1_2
    diag_u = cnot_1_2 @ np.kron(I, rz(-psi)) @ cnot_1_2

    a = _fix_global_phase(recon, U, a)

    a_1, a_2, a_3 = get_zyz_angles(a)
    b_1, b_2, b_3 = get_zyz_angles(b)
    c_1, c_2, c_3 = get_zyz_angles(c)
    d_1, d_2, d_3 = get_zyz_angles(d)

    two_cnot_unitary_gates = deque()
    two_cnot_unitary_gates.append(('RZ', c_3, 1))
    two_cnot_unitary_gates.append(('RY', c_2, 1))
    two_cnot_unitary_gates.append(('RZ', c_1, 1))
    two_cnot_unitary_gates.append(('RZ', d_3, source))
    two_cnot_unitary_gates.append(('RY', d_2, source))
    two_cnot_unitary_gates.append(('RZ', d_1, source))
    two_cnot_unitary_gates.append((1, source))
    two_cnot_unitary_gates.append(('RZ', phi, source))
    two_cnot_unitary_gates.append(('RX', theta + np.pi, 1))
    two_cnot_unitary_gates.append((1, source))
    two_cnot_unitary_gates.append(('RZ', a_3, 1))
    two_cnot_unitary_gates.append(('RY', a_2, 1))
    two_cnot_unitary_gates.append(('RZ', a_1, 1))
    two_cnot_unitary_gates.append(('RZ', b_3, source))
    two_cnot_unitary_gates.append(('RY', b_2, source))
    two_cnot_unitary_gates.append(('RZ', b_1, source))

    return diag_u * phase, two_cnot_unitary_gates

def three_cnot_decomposition(u, source):
    U, _ = project_to_SU4(u)
    gamma_U = gamma_map(U)
    eigvals = np.linalg.eigvals(gamma_U)
    angles = _robust_angle_sort(eigvals)

    alpha = -(angles[0] + angles[1]) / 2 - np.pi / 2
    beta = (angles[0] + angles[2]) / 2 + np.pi / 2
    delta = -(angles[1] + angles[2]) / 2 - np.pi / 2

    kernel = cnot_2_1 @ np.kron(I, ry(alpha)) @ cnot_1_2 @ np.kron(rz(delta), ry(beta)) @ cnot_2_1

    U_E = (E_dgr @ U @ E)
    k_E = (E_dgr @ kernel @ E)

    a, b, c, d = get_single_qubit_unitaries(U_E, k_E)

    recon = np.kron(a, b) @ kernel @ np.kron(c, d)
    
    a = _fix_global_phase(recon, U, a)

    a_1, a_2, a_3 = get_zyz_angles(a)
    b_1, b_2, b_3 = get_zyz_angles(b)
    c_1, c_2, c_3 = get_zyz_angles(c)
    d_1, d_2, d_3 = get_zyz_angles(d)

    three_cnot_unitary_gates = deque()
    three_cnot_unitary_gates.append(('RZ', c_3, 1))
    three_cnot_unitary_gates.append(('RY', c_2, 1))
    three_cnot_unitary_gates.append(('RZ', c_1, 1))
    three_cnot_unitary_gates.append(('RZ', d_3, source))
    three_cnot_unitary_gates.append(('RY', d_2, source))
    three_cnot_unitary_gates.append(('RZ', d_1, source))
    three_cnot_unitary_gates.append((source, 1))
    three_cnot_unitary_gates.append(('RZ', delta, 1))
    three_cnot_unitary_gates.append(('RY', beta, source))
    three_cnot_unitary_gates.append((1, source))
    three_cnot_unitary_gates.append(('RY', alpha, source))
    three_cnot_unitary_gates.append((source, 1))
    three_cnot_unitary_gates.append(('RZ', a_3, 1))
    three_cnot_unitary_gates.append(('RY', a_2, 1))
    three_cnot_unitary_gates.append(('RZ', a_1, 1))
    three_cnot_unitary_gates.append(('RZ', b_3, source))
    three_cnot_unitary_gates.append(('RY', b_2, source))
    three_cnot_unitary_gates.append(('RZ', b_1, source))

    return three_cnot_unitary_gates
