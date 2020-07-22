#=
project_jul:
- Julia version: 1.0.4
- Author: elyco
- Date: 2020-07-19
=#
include("RK.jl")
using LinearAlgebra
using Distributions
using Random
using Debugger
using JLD2
using Formatting: format
using Dates
using PyCall
pygui(:qt5)
using PyPlot
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = "14"

include("StructModule.jl")
using .StructModule

# using Distributed
@everywhere begin
    # using Distributed
    # include("StructModule.jl")
    # using .StructModule
    using LinearAlgebra: norm
    using SharedArrays
    THETA = 1
    GRAVITATIONAL_CONSTANT = 4.30091e-3
    SOFT_PARAM = 1e3  # Parsec
end

# constants
N_INITIAL = 500  # 5000

M_TOT = 1e11  # Sun masses
R_INITIAL = 50e3  # Parsec
SOFT_PARAM = 1e3  # Parsec
MAX_BOX_SIZE = 666e3  # Parsec
T_FINAL = 20e9  # years, = 20453 in units time of the exercise
T_FINAL = 20453  # in units time of the exercise
STEP_T = T_FINAL/20e4
THETA = 1
GRAVITATIONAL_CONSTANT = 4.30091e-3


function build_tree!(system::MassSystem)
    borders = get_current_box_size(system)  # x_lim, y_lim, z_lim

    root = Node(borders)
    # println(system.N)
    root.masses_indices = collect(1:system.N)
    root.center_of_mass = mean(system.positions, dims=1)[1,:]
    root.mass_count = system.N

    build_tree_helper!(system, root)
    return root
end

function build_tree_helper!(system::MassSystem, node::Node)
    xlim = node.borders[:,1]
    ylim = node.borders[:,2]
    zlim = node.borders[:,3]
    middle_limits = [mean(lim,dims=1) for lim in (xlim,ylim,zlim)]

    borders8 = [[sort!([xlim[i], middle_limits[1][1]]),
                 sort!([ylim[j], middle_limits[2][1]]),
                 sort!([zlim[k], middle_limits[3][1]])]
                for i in 1:2 for j in 1:2 for k in 1:2]

    borders8 = [hcat(b[1],b[2],b[3]) for b in borders8]

    leafs = []
    for border in borders8
        leaf = Node(border)
        leaf.parent = node
        fill_attributes!(system,leaf)
        if leaf.mass_count > 0
            push!(leafs, leaf)
            if leaf.mass_count > 1
                build_tree_helper!(system,leaf)
            end
        end
    end
    node.leafs = leafs
end

function fill_attributes!(system, node)
    masses_indices = node.parent.masses_indices[:]
    for i in masses_indices
        point = system.positions[i,:]
        if point_in_box(point, node.borders)
            push!(node.masses_indices,i)
            remove!(node.parent.masses_indices,i)
        end
    end
    if length(node.masses_indices) > 0
        node.center_of_mass = mean(system.positions[node.masses_indices, :], dims=1)[:]
        node.mass_count = length(node.masses_indices)
    end
end


@everywhere function point_in_box(point, borders)
    return borders[1,1] <= point[1] < borders[2,1] &&
           borders[1,2] <= point[2] < borders[2,2] &&
           borders[1,3] <= point[3] < borders[2,3]
end


function remove!(a, item)
    deleteat!(a, findall(x->x==item, a))
end


function get_current_box_size(system::MassSystem)
    x_lim = [minimum(system.positions[:, 1])-1, maximum(system.positions[:, 1])+1]
    y_lim = [minimum(system.positions[:, 2])-1, maximum(system.positions[:, 2])+1]
    z_lim = [minimum(system.positions[:, 3])-1, maximum(system.positions[:, 3])+1]
    return hcat(x_lim, y_lim, z_lim)
end


function calculate_force!(system, energy=false)
    """
    Initiate the calculation of the force for each point mass we are saving the force that act on it.
    :return:
    """
    total_energy = 0.
    if energy
        i = 1
        for point in eachrow(system.positions)
            total_energy = total_energy + calculate_force_helper!(system, system.root, point,energy)
            i = i + 1
        end
        return total_energy
    end

    # addprocs(8)
    tmp = convert(SharedArray{Float64,2},zeros(system.N, 3))
    @sync begin
        # pmap(i-> (
        #     point = system.positions[i,:];
        #     tmp[i,:] = calculate_force_helper!(system, system.root, point)
        #     ), 1:system.N)
        @distributed for i in 1:system.N
            point = system.positions[i,:]
            tmp[i,:] = calculate_force_helper!(system, system.root, point)
        end
    end
   system.forces = tmp
    # system.forces = zeros(system.N, 3)
    # i = 1
    # for point in eachrow(system.positions)
    #     system.forces[i,:] = calculate_force_helper!(system, system.root, point, energy)
    #     i = i + 1
    # end
end


@everywhere function calculate_force_helper!(system, node, point, energy=false)
    """
    Recursive function return the for acting on "point" from all the masses in "node"
    """
    force = [0., 0., 0.]
    if energy
        force = 0.
    end
    if node.mass_count == 0
        # exit condition from the recursion
        return force
    end

    # define the vector between two point
    distance_vec = .-(point .- node.center_of_mass)  # attractive force
    distance = norm(distance_vec)

    if node.mass_count == 1
        # if just 1 mass so the force is simply the force between them
        if distance == 0
            # unless we are calculating for the same point
            return force  # [0., 0., 0.]
        end
        # compute and return the force
        if energy
            return .- GRAVITATIONAL_CONSTANT * system.each_mass ^ 2 / (distance + SOFT_PARAM)
        end

        force_amplitude = GRAVITATIONAL_CONSTANT * system.each_mass ^ 2 / (distance + SOFT_PARAM) ^ 2
        force_direction = distance_vec / distance
        return force_amplitude * force_direction
    else
        # mass_count >= 2
        if distance / node.diagonal < THETA || point_in_box(point, node.borders)
            # if too close we are need to get inside the recursion

            for leaf in node.leafs
                force = force + calculate_force_helper!(system, leaf, point, energy)
            end
        else
            if energy
                return .- GRAVITATIONAL_CONSTANT * system.each_mass ^ 2 / (distance + SOFT_PARAM)
            end
            # we don't need to go further just multiply the force by the number of masses inside this node
            force_amplitude = node.mass_count * GRAVITATIONAL_CONSTANT * system.each_mass ^ 2 / (distance + SOFT_PARAM) ^ 2
            force_direction = distance_vec / distance
            return force_amplitude * force_direction
        end
    end

    # I don't think this line is ever executed
    return force
end


function set_values_from_flat!(y,system)
    dof = system.N * 3
    system.positions = reshape(y[1:dof], (system.N, 3))
    system.velocities = reshape(y[dof+1:end], (system.N, 3))
end


function get_current_values(system)
    r = system.positions[:]
    v = system.velocities[:]
    vcat(r, v)
end

function ode_to_solve_my_RK(t,y,system)
    dof = system.N * 3
    set_values_from_flat!(y,system)
    system.root = build_tree!(system)
    calculate_force!(system)

    drdt = y[dof+1:end]
    dvdt = system.forces[:] ./ system.each_mass
    # println(dvdt)
    dy = vcat(drdt, dvdt)
end


function remove_exceeds_masses(system)
    ind_to_del = maximum((system.positions .< -MAX_BOX_SIZE) .| (system.positions .> MAX_BOX_SIZE), dims=2)[:]
    ind_to_del = (1:system.N)[ind_to_del]
    if length(ind_to_del) > 0
        system.N = system.N - length(ind_to_del)
        system.positions  =  system.positions[setdiff(1:end, ind_to_del),:]
        vel_ind_to_del  = system.velocities[ind_to_del,:]
        system.velocities = system.velocities[setdiff(1:end, ind_to_del),:]
        return sum(0.5 .* system.each_mass .* (vel_ind_to_del[:,1] .^ 2 + vel_ind_to_del[:,2] .^ 2 + vel_ind_to_del[:,3] .^ 2))
    else
        return 0.
    end
end


function scatter_(mass_system::MassSystem, cam)
    p = mass_system.positions
    scatter(p[:,1],p[:,2],p[:,3], camera = cam)
    xlm = ylm = zlm = (-MAX_BOX_SIZE/3, MAX_BOX_SIZE/3)  # graph to reproduce the magnification from mousing
    xlims!(xlm[1], xlm[2])  # Reproduce magnification
    ylims!(ylm[1], ylm[2])
    zlims!(zlm[1], zlm[2])
end
function scatter_(mass_system::MassSystem)
    scatter_(mass_system, (40,50))
end

function calculate_energy(mass_system::MassSystem)
    potential_energy = calculate_force!(mass_system, true)
    v_squre = mass_system.velocities[:,1] .^ 2 + mass_system.velocities[:,2] .^ 2 + mass_system.velocities[:,3] .^ 2
    kinectic_energy = sum(0.5 .* mass_system.each_mass .* v_squre)
    [kinectic_energy, potential_energy]

end

function save_figures(num, save_positions, folder)
    # ioff()
    fig = figure(num)
    ax = fig.add_subplot(111, projection="3d")
    points = save_positions[1][1:Int(end/2),:]
    scat = ax.scatter3D(points[:,1],points[:,2], points[:,3])
    xlm = ylm = zlm = (-MAX_BOX_SIZE/3, MAX_BOX_SIZE/3)  # graph to reproduce the magnification from mousing
    ax.set_xlim3d(xlm[1], xlm[2])  # Reproduce magnification
    ax.set_ylim3d(ylm[1], ylm[2])  # ...
    ax.set_zlim3d(zlm[1], zlm[2])  #
    i = 0
    for points in save_positions
        p = points[1:Int(end/2),:]
        scat.remove()
        scat = ax.scatter3D(p[:,1],p[:,2], p[:,3], color="C0")
        PyPlot.savefig(string(folder,i,".png"), bbox_inches="tight", dpi=100)
        i = i+1
    end
end


py"""
def gif_(folder, name):
    import imageio
    import os
    import re

    filename = [fn for fn in os.listdir(folder) if fn.endswith(".png")]
    filename.sort(key=lambda f: int(re.sub("\D", "", f)))
    images = [imageio.imread(folder + fn) for fn in filename]

    imageio.mimsave(folder + name + ".gif", images)
"""

function create_folder()
    n = Dates.now()
    now = Dates.now()
    day = Dates.format(n, "mm_dd")
    hour = Dates.format(n, "HH_MM_SS")
    if ~ispath(string("./results/",day))
        mkdir(string("./results/",day))
    end
    if ~ispath(string("./results/",day,"/",hour,"/"))
        mkdir(string("./results/",day,"/",hour,"/"))
    end
    string("./results/",day,"/",hour,"/")
end


function start_cal(n, mass_system)
    count_dead = 0
    save_positions = []
    energy = []
    energy_of_lost_massess = []
    for i in 1:n
        println(i)
        y_0 = get_current_values(mass_system)
        if length(y_0)/6 < N_INITIAL - count_dead
            count_dead = count_dead + 1
            println(count_dead, " particles are deads")
            # println(length(y_0))
            # println(length(mass_system.positions[:]))
            # println(length(mass_system.velocities[:]))
            # println(mass_system.N)
        end

        ode_solver = make_solver(ode_to_solve_my_RK, t_span, "RK4", abstol, mass_system)
        sol = ode_solver(y_0)[2][:,end]
        set_values_from_flat!(sol, mass_system)

        push!(energy, calculate_energy(mass_system) )
        push!(energy_of_lost_massess, remove_exceeds_masses(mass_system))

        if i % 2 == 0
            push!(save_positions,vcat(mass_system.positions,mass_system.velocities))
        end
        if mass_system.N == 0
            println("Simulation is over, no more particles")
            return save_positions, vcat(energy'...) , cumsum(energy_of_lost_massess)
        end
    end

    save_positions, vcat(energy'...) , cumsum(energy_of_lost_massess)
end

# define parameters of calculation
Random.seed!(123)
velocity = 80.
tf = 80.
n = floor(Int, T_FINAL/tf + 1)
t_span = (0., tf)
abstol = 100

# define the system
mass_system = MassSystem(N_INITIAL,velocity)
mass_system.root = build_tree!(mass_system)

# start simulation
# n=10
t = @elapsed all_positions,energy, lost_masses = start_cal(n, mass_system)

# make a folder for saving the results
folder = create_folder()
new_folder_rename = folder[1:end-1] * format("_eps_{}_N_mass_{}_repeat_ode_{}_v_{}_time_{:.2f}_m_jullia/",abstol,N_INITIAL,n,velocity,t/60)
mv(folder, new_folder_rename)
folder = new_folder_rename

# save results to files
@save folder*"all_pos.jld2" all_positions
# all_positions = @load "all_pos.jld2" or something like that
save_figures(1, all_positions, folder)
py"gif_"(folder,"animated")

# plot the graphs
time_series = (0:n-1) .* tf
total_energy1 = sum(energy, dims=2)
total_energy2 = sum(energy, dims=2) .+ lost_masses
using Plots
using LaTeXStrings
Plots.plot(time_series, -1 .+ total_energy1 ./ total_energy1[1], label=L"Energy Conservation")
Plots.plot!(time_series, -1 .+ total_energy2 ./ total_energy2[1], label=L"Energy\ Conservation\ with\ lost")
Plots.plot!(time_series, -2 .* energy[:,1] ./ energy[:,2],label=L"\dfrac{E_k}{E_p}")
Plots.plot!(time_series, -2 .* (energy[:,1] .+ lost_masses) ./ energy[:,2],label=L"\dfrac{E_k}{E_p}+lost\ masses")
xlabel!("t"*" [astrnumical units]")

Plots.savefig(folder*"plot.png")
