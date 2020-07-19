#=
project_jul:
- Julia version: 1.0.4
- Author: elyco
- Date: 2020-07-19
=#
using LinearAlgebra
using Random, Distributions

# constants
N = 5000  # 5000
M_TOT = 1e11  # Sun masses
R_INITIAL = 50e3  # Parsec
SOFT_PARAM = 1e3  # Parsec
MAX_BOX_SIZE = 666e3  # Parsec
T_FINAL = 20e9  # years, = 20453 in units time of the exercise
T_FINAL = 20453  # in units time of the exercise
STEP_T = T_FINAL/20e4
THETA = 1
GRAVITATIONAL_CONSTANT = 4.30091e-3


mutable struct Node
    borders::Array{Float64,2}
    diagonal::Float64
    count_mass::Int64
    leafs::Array{Node,1}
    parent::Node
    center_of_mass::Array{Float64,1}
    masses_indices::Array{Int32,1}
    function Node(borders_::Array{Float64,2}, count_mass_::Int32)
        x = new();
        x.borders = borders_;
        x.diagonal = norm(borders_[1,:]-borders_[2,:], 2);
        x.count_mass = count_mass_;
        x.center_of_mass = zeros(3)
        x.masses_indices = []
        x
    end
end
function Node(borders::Array{Float64,2})
   return Node(borders,0)
end

mutable struct MassSystem
    N::Int32
    each_mass::Float64
    positions::Array{Float64,2}
    velocities::Array{Float64,2}
    forces::Array{Float64,2}
    root::Node
end
function MassSystem(N::Int32,velocity::Float64)
    each_mass = M_TOT/N

    # calculate positions
    u = rand(N)
    cos_theta = rand(N)
    phi = rand(N)

    phi = 2*pi*phi
    cos_theta = 2 .* cos_theta .- 1
    sin_theta = sqrt.(1 .- cos_theta .^ 2)
    r = R_INITIAL .* u .^ (1/3)
    positions = r .* [sin_theta .* cos.(phi) sin_theta .* sin.(phi) cos_theta]

    d = Normal(0, velocity/sqrt(3) )
    # calculate velocities
    velocities = rand(d, N, 3)
    # build tree
    root = build_tree!(self)
    root = Node(zeros(2,3))
    forces = zeros(3,N)
    MassSystem(N,each_mass,positions,velocities,forces, root)
end

a = MassSystem(10,40.)
v = a.velocities
p = a.positions

function build_tree!(system::MassSystem)
    borders = get_current_box_size(system)  # x_lim, y_lim, z_lim

    root = Node(borders)
    root.masses_indices = collect(1:N)
    root.center_of_mass = mean(system.positions, 0)
    root.mass_count = system.N

    build_tree_helper!(system, root)
    return root
end

function build_tree_helper!(system::MassSystem, node::Node)
    middle_limits = [mean(lim) for lim in node.borders]
    x_lim, y_lim, z_lim = node.borders

    borders8 = [[sorted([x_lim[i], middle_limits[0]]),
                 sorted([y_lim[j], middle_limits[1]]),
                 sorted([z_lim[k], middle_limits[2]])]
                for i in range(2) for j in range(2) for k in range(2)]
    leafs = []
    for border in borders8:
        leaf = Node(array(border), father=node)
        fill_attributes(system,leaf)
        if leaf.mass_count > 0:
            leafs.append(leaf)
            if leaf.mass_count > 1:
                build_tree_helper(system,leaf)
    node.leafs = leafs
end

function get_current_box_size(system::MassSystem)
    x_lim = [minimum(system.positions[:, 1])-1, maximum(system.positions[:, 1])+1]
    y_lim = [minimum(system.positions[:, 2])-1, maximum(system.positions[:, 2])+1]
    z_lim = [minimum(system.positions[:, 3])-1, maximum(system.positions[:, 3])+1]
    return [x_lim, y_lim, z_lim]
end
