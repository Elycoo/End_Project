import numpy as np
from scipy.integrate import solve_ivp

from matplotlib import pyplot as plt
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D  # <--- This is important for 3d plotting
# import matplotlib
# matplotlib.use('Qt5Agg')

import project_helper
from runga_kuta import make_solver
from numba import jit, deferred_type, optional, float64, int16
import numba
from numba.experimental import jitclass
from pytictoc import TicToc
import os

plt.rcParams["font.size"] = "14"

# constant
N_INITIAL = 50  # 5000
M_TOT = 10 ** 11  # Sun masses
R_INITIAL = 50e3  # Parsec
SOFT_PARAM = 1e3  # Parsec
MAX_BOX_SIZE = 666e3  # Parsec
T_FINAL = 20e9  # years, = 20453 in units time of the exercise
T_FINAL = 20453  # in units time of the exercise
STEP_T = T_FINAL / 20e4
THETA = 1
GRAVITATIONAL_CONSTANT = 4.30091e-3

node_type = deferred_type()

spec = [
    ('leafs', optional(node_type)[:]),
    ('father', optional(node_type)),
    ('borders', float64[:,:]),
    ('center_of_mass', float64[:]),
    ('masses_indices', int16[:]),
    ('mass_count', int16),
    ('main_diagonal', float64),
]


# @jitclass(spec)
class Node:
    """
    Tree structure for saving all the data in the mass system
    """

    def __init__(self, borders, father=None):
        # leaf are the children of some node
        self.father = father

        # initialize the basics attributes
        self.borders = borders
        self.center_of_mass = np.array([0., 0., 0.])
        self.masses_indices = np.arange(0,dtype=np.int16)
        self.masses_indices = []
        self.mass_count = 0

        # we can compute already the main diagonal of the box
        two_point = self.borders.T
        self.main_diagonal = np.sqrt(np.sum((two_point[0] - two_point[1])*(two_point[0] - two_point[1])))

# node_type.define(Node.class_type.instance_type)


def remove_exceeds_masses(positions, velocities, N):
    ind_to_del = np.argwhere((positions < -MAX_BOX_SIZE) | (positions > MAX_BOX_SIZE))[:, 0]
    if len(ind_to_del) > 0:
        ind_to_del = np.unique(ind_to_del, axis=0)
    N = N - len(ind_to_del)
    positions = np.delete(positions, ind_to_del, axis=0)
    velocities = np.delete(velocities, ind_to_del, axis=0)
    return positions, velocities, N


def get_current_values(positions, velocities):
    r = positions.flatten()
    v = velocities.flatten()
    return np.hstack((r, v)).reshape((-1,1))


# @jit
def get_values_from_flat(N, y):
    dof = int(N * 3)
    positions = y[:dof].reshape((N, 3))
    velocities = y[dof:].reshape((N, 3))
    return positions, velocities


# @jit
def build_tree(positions, N):
    borders = get_current_box_size(positions)  # x_lim, y_lim, z_lim

    root = Node(borders)
    root.masses_indices = list(range(N))
    root.center_of_mass = np.mean(positions, axis=0)
    root.mass_count = N

    build_tree_helper(root, positions)
    return root


# @njit
def build_tree_helper(node, positions):
    middle_limits = [np.mean(lim) for lim in node.borders]
    x_lim, y_lim, z_lim = node.borders

    borders8 = [[sorted([x_lim[i], middle_limits[0]]),
                 sorted([y_lim[j], middle_limits[1]]),
                 sorted([z_lim[k], middle_limits[2]])]
                for i in range(2) for j in range(2) for k in range(2)]
    leafs = []
    for border in borders8:
        leaf = Node(np.array(border), father=node)
        fill_attributes(leaf, positions)
        if leaf.mass_count > 0:
            leafs.append(leaf)
            if leaf.mass_count > 1:
                build_tree_helper(leaf, positions)
    node.leafs = leafs


# @jit
def fill_attributes(node, positions):
    masses_indices = node.father.masses_indices[:]
    for i in masses_indices:
        point = positions[i]
        if point_in_box(point, node.borders):
            node.masses_indices.append(i)
            node.father.masses_indices.remove(i)
    if len(node.masses_indices) > 0:
        node.center_of_mass = np.mean(positions[node.masses_indices, :], axis=0)
        node.mass_count = len(node.masses_indices)


# @jit
def get_current_box_size(positions):
    x_lim = np.array([np.min(positions[:, 0]) - 1, np.max(positions[:, 0]) + 1])
    y_lim = np.array([np.min(positions[:, 1]) - 1, np.max(positions[:, 1]) + 1])
    z_lim = np.array([np.min(positions[:, 2]) - 1, np.max(positions[:, 2]) + 1])
    return np.array([x_lim, y_lim, z_lim])


@jit
def calculate_force(positions, root, N):
    """
    Initiate the calculation of the force for each point mass we are saving the force that act on it.
    :return:
    """
    forces = np.zeros((N, 3))
    for i, point in enumerate(positions):
        forces[i] = calculate_force_helper(root, point)

    return forces


@jit
def calculate_force_helper(node, point):
    """
    Recursive function return the for acting on "point" from all the masses in "node"
    """
    force = np.array([0., 0., 0.])
    if node.mass_count == 0:
        # exit condition from the recursion
        return force

    # define the vector between two point
    distance_vec = -(point - node.center_of_mass)  # attractive force
    distance = np.linalg.norm(distance_vec)

    if node.mass_count == 1:
        # if just 1 mass so the force is simply the force between them
        if distance == 0:
            # unless we are calculating for the same point
            return force  # np.array([0., 0., 0.])

        # compute and return the force
        force_amplitude = GRAVITATIONAL_CONSTANT * each_mass ** 2 / (distance + SOFT_PARAM) ** 2
        force_direction = distance_vec / distance
        return force_amplitude * force_direction
    else:
        # mass_count >= 2
        if distance / node.main_diagonal < THETA or point_in_box(point, node.borders):
            # if too close we are need to get inside the recursion
            for leaf in node.leafs:
                force = force + calculate_force_helper(leaf, point)
        else:
            # we don't need to go further just multiply the force by the number of masses inside this node
            force_amplitude = node.mass_count * GRAVITATIONAL_CONSTANT * each_mass ** 2 / (
                        distance + SOFT_PARAM) ** 2
            force_direction = distance_vec / distance
            return force_amplitude * force_direction

    # I don't think this line is ever execute
    return force


# @jit
def ode_to_solve(t, y):
    global N
    dof = int(N * 3)
    positions_tmp, _ = get_values_from_flat(N, y)
    root = build_tree(positions_tmp, N)
    forces = calculate_force(positions_tmp, root, N)

    drdt = y[dof:]
    dvdt = forces.flatten() / each_mass
    return np.hstack((drdt, dvdt))


@jit
def point_in_box(point, borders):
    return borders[0][0] <= point[0] < borders[0][1] and \
           borders[1][0] <= point[1] < borders[1][1] and \
           borders[2][0] <= point[2] < borders[2][1]


# @jit
def start_cal(n):
    count_dead = 0
    global positions, velocities, N
    save_positions = [positions]
    for i in range(n):
        print(i)
        if i == 2:
            tic.toc(restart=True)
        y_0 = get_current_values(positions, velocities)
        if y_0.size//6 < N_INITIAL - count_dead:
            count_dead = count_dead + 1
            # print('The len of y_0/6 is ' + str(len(y_0) // 6))
            print(str(count_dead) + ' particles are deads')
        _, sol = my_solver(y_0)
        # sol = solve_ivp(ode_to_solve(mass_system), t_span, y_0, rtol=0.1, atol=eps)
        positions, velocities = get_values_from_flat(N, sol[:, -1])
        positions, velocities, N = remove_exceeds_masses(positions, velocities, N)
        if i % 2 == 0:
            save_positions.append(positions)
        if N == 0:
            return save_positions
    return save_positions


if __name__ == '__main__':
    np.random.seed(123)
    """
    Object that have all the properties of the system.
    It save the positions, velocities and have also the root of the Tree.
    Some of the important methods are:
    build_tree, calculate_force ode_to_solve
    """
    # each mass is the total mass divided by the number of the point_like mass
    velocity = 80

    N = N_INITIAL
    each_mass = M_TOT / N_INITIAL

    # calculate positions
    u, cos_theta, phi = np.random.rand(3, N)

    phi = 2 * np.pi * phi
    cos_theta = 2 * cos_theta - 1
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    r = R_INITIAL * u ** (1 / 3)

    positions = (r * np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])).T

    # calculate velocities
    velocities = np.random.normal(0, velocity / np.sqrt(3), (N, 3))
    # build tree

    root = build_tree(positions, N)
    # calculate forces
    forces = np.zeros_like(positions)
    count_img = 0

    forces = calculate_force(positions, root, N)

    plt.close('all')
    tic = TicToc()

    # define parameters of calculation
    y_0 = get_current_values(positions, velocities)
    # t_span = (0, T_FINAL / n)
    tf = 80
    n = T_FINAL // tf + 1
    n = 10
    t_span = (0, tf)
    eps = 0.1

    my_solver = make_solver(ode_to_solve, t_span, 'RK4', eps)

    # start simulation
    tic.tic()
    all_positions = start_cal(n)
    tic.toc()
    time = tic.tocvalue()
    folder = project_helper.create_folder()
    np.save(folder + 'all_pos', all_positions)

    v = velocity
    new_folder_rename = folder[:-1] + f'_eps_{eps}_N_mass_{N}_{N_INITIAL}_repeat_ode_{n}_v_{v}_time_{time / 60:.2f}_m_numba_var.py_/'
    if '@jit'.startswith('@'):
        new_folder_rename = new_folder_rename[:-1] + 'jit/'
    os.rename(folder, new_folder_rename)

    # all_positions = list(np.load('all_pos.npy', allow_pickle=True))
    tic.tic()
    project_helper.save_figures(2, all_positions, new_folder_rename)
    project_helper.gif(new_folder_rename, 'animate')
    tic.toc()

    project_helper.beep()
