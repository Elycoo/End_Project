import numpy as np
from scipy.integrate import solve_ivp

from matplotlib import pyplot as plt
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D  # <--- This is important for 3d plotting
# import matplotlib
# matplotlib.use('Qt5Agg')

import project_helper
from runga_kuta import make_solver
from pytictoc import TicToc
import os

plt.rcParams["font.size"] = "14"

# constant
N = 2  # 5000
M_TOT = 10**11  # Sun masses
R_INITIAL = 50e3  # Parsec
SOFT_PARAM = 1e3  # Parsec
MAX_BOX_SIZE = 666e3  # Parsec
T_FINAL = 20e9  # years, = 20453 in units time of the exercise
T_FINAL = 20453  # in units time of the exercise
STEP_T = T_FINAL/20e4
THETA = 1
GRAVITATIONAL_CONSTANT = 4.30091e-3

class Node:
    """
    Tree structure for saving all the data in the mass system
    """
    def __init__(self, borders, father=None):
        # leaf are the children of some node
        self.leafs = None
        self.father = father

        # initialize the basics attributes
        self.borders = borders
        self.center_of_mass = np.array([0., 0., 0.])
        self.masses_indices = []
        self.mass_count = 0

        # we can compute already the main diagonal of the box
        two_point = self.borders.T
        self.main_diagonal = np.linalg.norm(two_point[0] - two_point[1])

    def is_end_leaf(self):
        # not used
        return self.leafs is None


class MassSystem():
    """
    Object that have all the properties of the system.
    It save the positions, velocities and have also the root of the Tree.
    Some of the important methods are:
    build_tree, calculate_force ode_to_solve
    """
    def __init__(self, velocity):
        # each mass is the total mass divided by the number of the point_like mass
        self.N = N
        self.each_mass = M_TOT/self.N

        # calculate positions
        u, cos_theta, phi = np.random.rand(3, self.N)

        phi = 2*np.pi*phi
        cos_theta = 2*cos_theta - 1
        sin_theta = np.sqrt(1-cos_theta**2)
        r = R_INITIAL * u**(1/3)

        self.positions = (r * np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])).T

        # calculate velocities
        self.velocities = np.random.normal(0, velocity / np.sqrt(3), (self.N, 3))
        # build tree
        self.root = build_tree(self)
        # calculate forces
        self.forces = np.zeros_like(self.positions)
        self.count_img = 0

    def remove_exceeds_masses(self):
        ind_to_del = np.argwhere((mass_system.positions < -MAX_BOX_SIZE) | (mass_system.positions > MAX_BOX_SIZE))[:, 0]
        if len(ind_to_del) > 0:
            ind_to_del = np.unique(ind_to_del, axis=0)
        self.N = self.N - len(ind_to_del)
        self.positions = np.delete(self.positions, ind_to_del, axis=0)
        self.velocities = np.delete(self.velocities, ind_to_del, axis=0)

    def get_current_values(self):
        r = self.positions.flatten()
        v = self.velocities.flatten()
        return np.hstack((r, v)).reshape((-1,1))

    def set_values_from_flat(self, y):
        dof = int(self.N * 3)
        self.positions = np.reshape(y[:dof], (self.N, 3))
        self.velocities = np.reshape(y[dof:], (self.N, 3))

    def plot(self, num, show=False, save=True, camera_pos=False):
        """
        Plot the current state of the system in 3D figure plot
        """
        fig = plt.figure(num)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*self.positions.T)
        if show:
            plt.show()
            # fig.canvas.draw()
            # fig.canvas.flush_events()
        if save:
            plt.savefig(folder + f'{self.count_img}.png', bbox_inches='tight', dpi=100)
            self.count_img += 1


def build_tree(system):
    borders = get_current_box_size(system)  # x_lim, y_lim, z_lim

    root = Node(borders, father=system)
    root.masses_indices = list(range(system.N))
    root.center_of_mass = np.mean(system.positions, axis=0)
    root.mass_count = system.N

    build_tree_helper(system, root)
    return root


def build_tree_helper(system, node):
    middle_limits = [np.mean(lim) for lim in node.borders]
    x_lim, y_lim, z_lim = node.borders

    borders8 = [[sorted([x_lim[i], middle_limits[0]]),
                 sorted([y_lim[j], middle_limits[1]]),
                 sorted([z_lim[k], middle_limits[2]])]
                for i in range(2) for j in range(2) for k in range(2)]
    leafs = []
    for border in borders8:
        leaf = Node(np.array(border), father=node)
        fill_attributes(system,leaf)
        if leaf.mass_count > 0:
            leafs.append(leaf)
            if leaf.mass_count > 1:
                build_tree_helper(system,leaf)
    node.leafs = leafs


def fill_attributes(system, node):
    masses_indices = node.father.masses_indices[:]
    for i in masses_indices:
        point = system.positions[i]
        if point_in_box(point, node.borders):
            node.masses_indices.append(i)
            node.father.masses_indices.remove(i)
    if len(node.masses_indices) > 0:
        node.center_of_mass = np.mean(system.positions[node.masses_indices, :], axis=0)
        node.mass_count = len(node.masses_indices)


def get_current_box_size(system):
    x_lim = np.array([np.min(system.positions[:, 0])-1, np.max(system.positions[:, 0])+1])
    y_lim = np.array([np.min(system.positions[:, 1])-1, np.max(system.positions[:, 1])+1])
    z_lim = np.array([np.min(system.positions[:, 2])-1, np.max(system.positions[:, 2])+1])
    return np.array([x_lim, y_lim, z_lim])


def calculate_force(system):
    """
    Initiate the calculation of the force for each point mass we are saving the force that act on it.
    :return:
    """
    system.forces = np.zeros((system.N, 3))
    for i, point in enumerate(system.positions):
        system.forces[i] = calculate_force_helper(system, system.root, point)


def calculate_force_helper(system, node, point):
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
        force_amplitude = GRAVITATIONAL_CONSTANT * system.each_mass ** 2 / (distance + SOFT_PARAM) ** 2
        force_direction = distance_vec / distance
        return force_amplitude * force_direction
    else:
        # mass_count >= 2
        if distance / node.main_diagonal < THETA or point_in_box(point, node.borders):
            # if too close we are need to get inside the recursion
            for leaf in node.leafs:
                force = force + calculate_force_helper(system, leaf, point)
        else:
            # we don't need to go further just multiply the force by the number of masses inside this node
            force_amplitude = node.mass_count * GRAVITATIONAL_CONSTANT * system.each_mass ** 2 / (distance + SOFT_PARAM) ** 2
            force_direction = distance_vec / distance
            return force_amplitude * force_direction

    # I don't think this line is ever execute
    return force


def ode_to_solve(system, t, y):
    dof = int(system.N * 3)
    system.set_values_from_flat(y)
    system.root = build_tree(system)
    calculate_force(system)

    drdt = y[dof:]
    dvdt = system.forces.flatten()/system.each_mass
    return np.hstack((drdt, dvdt))


def point_in_box(point, borders):
    return borders[0][0] <= point[0] < borders[0][1] and\
           borders[1][0] <= point[1] < borders[1][1] and\
           borders[2][0] <= point[2] < borders[2][1]


def start_cal(n):
    save_positions = []
    try:
        for i in range(n):
            print(i)
            y_0 = mass_system.get_current_values()
            if len(y_0) < N*6:
                print(f'The len of y_0/6 is {len(y_0)//6}')
            _, sol = my_solver(y_0)
            # sol = solve_ivp(mass_system.ode_to_solve, t_span, y_0, rtol=0.1, atol=eps)
            mass_system.remove_exceeds_masses()
            if i % 2 == 0:
                save_positions.append(mass_system.positions)
            if mass_system.N == 0:
                return save_positions
    except KeyboardInterrupt:
        mass_system.remove_exceeds_masses()
        save_positions.append(mass_system.positions)
        print('KeyboardInterrupt')
    except Exception as e:
        # global emergency
        # emergency = save_positions
        raise e
    return save_positions


if __name__ == '__main__':
    np.random.seed(123)
    plt.close('all')
    tic = TicToc()

    # define the system
    v = 1
    mass_system = MassSystem(v)

    calculate_force(mass_system)
    forces_out = mass_system.forces

    # define parameters of calculation
    y_0 = mass_system.get_current_values()
    # t_span = (0, T_FINAL / n)
    tf = 80
    n = T_FINAL//tf + 1
    t_span = (0, tf)
    eps = 10

    def f(t, y):
        return ode_to_solve(mass_system, t, y)

    my_solver = make_solver(f, t_span, 'RK4', eps)

    # start simulation
    tic.tic()
    all_positions = start_cal(n)
    tic.toc()
    time = tic.tocvalue()

    # save results to files
    folder = project_helper.create_folder()
    np.save(folder + 'all_pos', all_positions)

    new_folder_rename = folder[:-1] + f'_eps_{eps}_N_mass_{N}_repeat_ode_{n}_v_{v}_time_{time/60:.2g}_m_regular.py_/'
    os.rename(folder, new_folder_rename)
    folder = new_folder_rename

    # all_positions = list(np.load('all_pos.npy', allow_pickle=True))
    tic.tic()
    project_helper.save_figures(2, all_positions, folder)
    project_helper.gif(folder, 'animate')
    tic.toc()

    project_helper.beep()
