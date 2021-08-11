# # Controller Design

# https://github.com/pydy/pydy-tutorial-human-standing/blob/master/notebooks/solution/control.py
from numpy import zeros, ones, matrix, eye, dot, asarray, matmul, hstack
from numpy.linalg import inv
from scipy.linalg import solve_continuous_are

def build_optimal_controller(kane, constants, equilibrium_dict):
    linear_state_matrix, linear_input_matrix, inputs = kane.linearize(new_method=True, A_and_B=True)
    f_A_lin = linear_state_matrix.subs(constants).subs(equilibrium_dict)
    f_B_lin = linear_input_matrix.subs(constants).subs(equilibrium_dict)
    m_mat = kane.mass_matrix_full.subs(constants).subs(equilibrium_dict)

    A = matrix(m_mat.inv() * f_A_lin).astype(float)
    B = matrix(m_mat.inv() * f_B_lin).astype(float)

    Q = matrix(eye(4))

    R = matrix(eye(2))

    S = solve_continuous_are(A, B, Q, R)

    K = inv(R) * B.T * S

    # This is an annoying little issue. We specified the order of things when
    # creating the rhs function, but the linearize function returns the F_B
    # matrix in the order corresponding to whatever order it finds the joint
    # torques. This would also screw things up if we specified a different
    # ordering of the coordinates and speeds as the standard kane._q + kane._u

    K = K[[1,0], :] # this is dumb haha

    C = hstack((eye((f_B_lin.shape[1])),zeros((f_B_lin.shape[1],f_B_lin.shape[1]))))

    D = zeros((f_B_lin.shape[1], f_B_lin.shape[1]))

    # N = (C(-A+BK)^-1 B)^-1
    Nbar = inv(matmul(matmul(C,inv(-A+matmul(B,K))),B))

    return K, Nbar

