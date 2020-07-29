# using Distributed

@everywhere module StructModule
using Distributions: Normal
using LinearAlgebra: norm, cross

export Node, MassSystem

const M_TOT = 1e11  # Sun masses
const R_INITIAL = 50e3  # Parsec
# R_INITIAL = 500.  # Parsec

mutable struct Node
    borders::Array{Float64,2}
    diagonal::Float64
    mass_count::Int64
    leafs::Array{Node,1}
    parent::Node
    center_of_mass::Array{Float64,1}
    masses_indices::Array{Int64,1}
    function Node(borders_::Array{Float64,2}, mass_count::Int64)
        x = new();
        x.borders = borders_;
        x.diagonal = norm(borders_[1,:]-borders_[2,:], 2);
        x.mass_count = mass_count;
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
function MassSystem(N::Int64,velocity::Float64)
    # initialize MassSystem
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
    if N == 2
        positions = rand()*[0 0 -R_INITIAL;0 0 R_INITIAL]
        velocities = [0 velocity 0;0 -velocity 0]
        println("change v, p")
    else
        velocities = rand(d,N,3)
    end

    # initialize forces
    forces = zeros(N,3)

    # build tree
    root = Node(zeros(2,3))
    system = MassSystem(N,each_mass,positions,velocities,forces,root)
    # system.root = build_tree!(system)
    system
end

end
