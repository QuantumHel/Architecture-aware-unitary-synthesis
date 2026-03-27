import numpy as np
from utils import orthogonal_congruence_diagonalize, get_zyz_angles, ry, rz, rx
from collections import deque
from qiskit.compiler import transpile
from qiskit_aer import Aer

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
    detU = np.linalg.det(U)
    if detU == 0: detU = 1e-15
    phase = detU ** (1 / 4)
    return U / phase, phase

def project_to_SU2(U):
    detU = np.linalg.det(U)
    if detU == 0: detU = 1e-15
    return U / detU ** (1 / 2)

def gamma_map(u):
    assert len(u) == 4
    return u @ sigma_y_kron_2 @ u.T @ sigma_y_kron_2

def extract_tensor_factors(M):
    M_reshaped = M.reshape(2, 2, 2, 2)
    M_reordered = M_reshaped.transpose(0, 2, 1, 3)
    M_flat = M_reordered.reshape(4, 4)
    
    # 2. SVD to extract the rank-1 tensor components
    u, s, vh = np.linalg.svd(M_flat)
    
    a = u[:, 0].reshape(2, 2)
    b = vh[0, :].reshape(2, 2)
    
    # 3. Restore the unitary scale (SVD vectors have norm 1, Unitaries have norm sqrt(2))
    a = a * np.sqrt(s[0])
    b = b * np.sqrt(s[0])
    
    # 4. Strip the arbitrary SVD complex phase to enforce strict SU(2) symmetry
    a = a / np.sqrt(np.linalg.det(a)) if np.linalg.det(a) != 0 else a / 1e-15
    b = b / np.sqrt(np.linalg.det(b)) if np.linalg.det(b) != 0 else b / 1e-15
    
    return a, b

def get_single_qubit_unitaries(U_E, k_E):
    S_U = U_E @ (U_E.T)
    S_k = k_E @ (k_E.T)

    A_U = orthogonal_congruence_diagonalize(S_U)
    B_k = orthogonal_congruence_diagonalize(S_k)

    if np.real(np.linalg.det(A_U @ B_k.T)) < 0:
            A_U[:, 0] = -A_U[:, 0]

    C = np.conjugate(k_E).T @ B_k @ A_U.T @ U_E

    A_tilde =  E @ A_U @ B_k.T @ E_dgr
    C_tilde = E @ C @ E_dgr

    a, b = extract_tensor_factors(A_tilde)
    c, d = extract_tensor_factors(C_tilde)

    return a, b, c, d

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

    angles = np.sort(np.angle(eigvals))


    theta = (angles[0] + angles[2]) / 2
    phi = (angles[0] - angles[2]) / 2

    U_E = (E_dgr @ U @ Delta @ E)

    kernel = cnot_1_2 @ np.kron(rx(theta + np.pi), rz(phi)) @ cnot_1_2 # Add + np.pi to get correct eigenvalues
    k_E = (E_dgr @ kernel @ E)

    a, b, c, d = get_single_qubit_unitaries(U_E, k_E)

    recon = np.kron(a, b) @ kernel @ np.kron(c, d) @ cnot_1_2 @ np.kron(I, rz(-psi)) @ cnot_1_2
    # recon = np.kron(-a, b) @ kernel @ np.kron(c, d)
    diag_u = cnot_1_2 @ np.kron(I, rz(-psi)) @ cnot_1_2

    phase_diff = np.trace(np.conjugate(recon).T @ U) / 4
    if np.real(phase_diff) < -0.5:
        a = -a

    # if not np.allclose(np.kron(a, b) @ kernel @ np.kron(c, d) @ cnot_1_2 @ np.kron(I, rz(-psi)) @ cnot_1_2, U):
    #     print(f"Unitary mismatch")

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
    angles = np.angle(eigvals)

    alpha = -(angles[0] + angles[1]) / 2 - np.pi / 2
    beta = (angles[0] + angles[2]) / 2 + np.pi / 2
    delta = -(angles[1] + angles[2]) / 2 - np.pi / 2

    kernel = cnot_2_1 @ np.kron(I, ry(alpha)) @ cnot_1_2 @ np.kron(rz(delta), ry(beta)) @ cnot_2_1

    U_E = (E_dgr @ U @ E)
    k_E = (E_dgr @ kernel @ E)

    a, b, c, d = get_single_qubit_unitaries(U_E, k_E)

    recon = np.kron(a, b) @ kernel @ np.kron(c, d)
    phase_diff = np.trace(np.conjugate(recon).T @ U) / 4
    
    if np.real(phase_diff) < -0.5:
        a = -a

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
