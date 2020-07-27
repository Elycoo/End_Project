#%%
# # # # # # # # import and constants # # # # # # # # # # # # # # # #
#=
project_jul:
- Julia version: 1.0.4
- Author: elyco
- Date: 2020-07-19
=#
using LinearAlgebra
using Distributions
using Random
using KernelDensity
using LsqFit

# using Debugger
using Plots
using LaTeXStrings
using Dates
using JSON
using Formatting: format

include("StructModule.jl")
using .StructModule
include("RK.jl")
include("helper.jl")

@everywhere begin
    using LinearAlgebra: norm
    using SharedArrays
    const THETA = 1
    const GRAVITATIONAL_CONSTANT = 4.30091e-3
    const SOFT_PARAM = 1e3  # Parsec
end

# constants
const N_INITIAL = 5000 # 5000

const M_TOT                   = 1e11  # Sun masses
const R_INITIAL               = 50e3  # Parsec
const SOFT_PARAM              = 1e3  # Parsec
const MAX_BOX_SIZE            = 666e3  # Parsec
# const T_FINAL                 = 20e9  # years, = 20453 in units time of the exercise
const T_FINAL                 = 20453  # in units time of the exercise
const THETA                   = 1
const GRAVITATIONAL_CONSTANT  = 4.30091e-3

#%% Tres construct

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

#%%
# force/energy calculation
"""
Initiate the calculation of the force for each point mass we are saving the force that act on it.
    :return:
"""
function calculate_force!(system)
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
    #     system.forces[i,:] = calculate_force_helper!(system, system.root, point)
    #     i = i + 1
    # end
end


@everywhere function calculate_force_helper!(system, node, point)
    """
    Recursive function return the for acting on "point" from all the masses in "node"
    """
    force = [0., 0., 0.]
    mass_count = node.mass_count

    if mass_count == 0
        # exit condition from the recursion
        return force
    end

    # define the vector between two point
    distance_vec = node.center_of_mass - point  # attractive force
    distance = norm(distance_vec)

    if mass_count == 1
        # if just 1 mass so the force is simply the force between them
        if distance == 0
            # unless we are calculating for the same point
            return force  # [0., 0., 0.]
        end
        # compute and return the force
        force_amplitude = GRAVITATIONAL_CONSTANT * system.each_mass ^ 2 / (distance + SOFT_PARAM) ^ 2
        force_direction = distance_vec / distance
        return force_amplitude * force_direction
    else
        # mass_count >= 2
        if distance / node.diagonal < THETA || point_in_box(point, node.borders)
            # if too close we are need to get inside the recursion

            for leaf in node.leafs
                force += calculate_force_helper!(system, leaf, point)
            end
            return force
        else
            # we don't need to go further just multiply the force by the number of masses inside this node
            force_amplitude = node.mass_count * GRAVITATIONAL_CONSTANT * system.each_mass ^ 2 / (distance + SOFT_PARAM) ^ 2
            force_direction = distance_vec / distance
            return force_amplitude * force_direction
        end
    end
end


function calculate_potential_energy(system)
    total_energy = 0.
    i = 1
    for point in eachrow(system.positions)
        total_energy += potential_energy_helper(system, system.root, point)
        i = i + 1
    end
    return total_energy
end


function potential_energy_helper(system, node, point)
    """
    Recursive function return the for acting on "point" from all the masses in "node"
    """
    energy = 0.

    if node.mass_count == 0
        # exit condition from the recursion
        return energy
    end

    # define the vector between two point
    distance_vec = .-(point .- node.center_of_mass)  # attractive energy
    distance = norm(distance_vec)

    if node.mass_count == 1
        # if just 1 mass so the energy is simply the energy between them
        if distance == 0
            # unless we are calculating for the same point
            return energy  # [0., 0., 0.]
        end
        # compute and return the energy
        return -GRAVITATIONAL_CONSTANT * system.each_mass ^ 2 / (distance + SOFT_PARAM)
    else
        # mass_count >= 2
        if distance / node.diagonal < THETA || point_in_box(point, node.borders)
            # if too close we are need to get inside the recursion

            for leaf in node.leafs
                energy += potential_energy_helper(system, leaf, point)
            end
            return energy
        else
            # we don't need to go further just multiply the energy by the number of masses inside this node
            return - GRAVITATIONAL_CONSTANT * system.each_mass ^ 2 / (distance + SOFT_PARAM)
        end
    end

end

function remove_exceeds_masses(system)
    ind_to_del = maximum((system.positions .< -MAX_BOX_SIZE) .| (system.positions .> MAX_BOX_SIZE), dims=2)[:]
    ind_to_del = (1:system.N)[ind_to_del]
    if length(ind_to_del) > 0
        system.N = system.N - length(ind_to_del)
        system.positions  = system.positions[setdiff(1:end, ind_to_del),:]
        system.velocities = system.velocities[setdiff(1:end, ind_to_del),:]
        return nothing
    else
        return nothing
    end
end

function calculate_energy(mass_system::MassSystem)
    potential_energy = calculate_potential_energy(mass_system)/2    # no double counting
    v_squre = mass_system.velocities[:,1] .^ 2 + mass_system.velocities[:,2] .^ 2 + mass_system.velocities[:,3] .^ 2
    kinectic_energy = 0.5 * mass_system.each_mass * sum(v_squre)
    [kinectic_energy, potential_energy]
end

#%%
function ode_to_solve_my_RK(t::Float64,y::Array{Float64,1},system::MassSystem)
    dof = system.N * 3
    set_values_from_flat!(y,system)
    system.root = build_tree!(system)
    calculate_force!(system)

    drdt = y[dof+1:end]
    dvdt = system.forces[:] ./ system.each_mass
    # println(dvdt)
    dy = vcat(drdt, dvdt)
end



function start_cal(n, mass_system, t_span, abstol)
    count_dead = 0
    save_positions = []
    energy = []
    energy_of_lost_massess = []
    kinetic_energy_of_lost_massess = []
    for i in 1:n
        println(i)
        y_0 = get_current_values(mass_system)
        if length(y_0)/6 < N_INITIAL - count_dead
            count_dead = count_dead + 1
            println(count_dead, " particles are deads")
        end

        ode_solver = make_solver(ode_to_solve_my_RK, t_span, "RK4", abstol, mass_system)
        sol = ode_solver(y_0)[2][:,end]
        set_values_from_flat!(sol, mass_system)

        energy_before = calculate_energy(mass_system)
        remove_exceeds_masses(mass_system)
        energy_after = calculate_energy(mass_system)
        energy_lost = energy_after .- energy_before

        push!(energy, energy_before)
        push!(energy_of_lost_massess, energy_lost )
        # push!(kinetic_energy_of_lost_massess, energy_lost[1])

        if i % 2 == 0
            push!(save_positions,mass_system.positions)
        end
        if mass_system.N == 0
            println("Simulation is over, no more particles")
            return save_positions, vcat(energy'...) , cumsum(energy_of_lost_massess)
        end
    end

    save_positions, vcat(energy'...) , vcat(cumsum(energy_of_lost_massess)'...)
end
#%%
function find_density(system::MassSystem)
    p = system.positions
    COM = system.root.center_of_mass
    radii =  sqrt.((p[:,1] .- COM[1]).^2 .+(p[:,2] .- COM[2]).^2 .+(p[:,3] .- COM[3]).^2)
    find_density(radii)
end
function find_density(radii::Array{Float64,1})
    xdata = range(minimum(radii), maximum(radii),length=10^3)
    xdata = range(minimum(radii), 10^5.2,length=10^3)
    p = kde(radii)
    ydata = KernelDensity.pdf(p,xdata)

    # curve_fit
    @. model(x, p) =  p[1]*x^2/(1 + (x/p[2])^p[3] )^(p[4]/p[3])
    # p0 = [1e-5,10^4.7,5.13,2.2]
    p0 = [1e-7, 39600, 1.14, 8.98]
    # p0=[6.52686738e-07, 3.10550454e+04, 2.50608032e+00, 4.79147796e+00]
    lb = [0 ,0.,0.,0.]  # lower bound
    fit = curve_fit(model, xdata, ydata, p0 ,lower=lb)
    println(fit.param)
    println(sum(fit.resid.^2))

    # plo = plot(xdata, ydata,xaxis=:log,yaxis=:log, marker="o", label="simulation")
    plo = plot(xdata,ydata, marker="o", label="simulation")
    plot!(xdata, model(xdata,fit.param), label="fit")

    # density
    # fit
end
R2(x::Float64,y::Float64,z::Float64) = x^2+y^2+z^2
R2(r::Array{Float64,1}) = r[1]^2+r[2]^2+r[3]^2
#%%
# define parameters of calculation
Random.seed!(1364)
velocity = 60.
# mass_system = MassSystem(N_INITIAL,velocity)

# for velocity in [80.] #[65., 80., 95.]

tf = 80.
n = ceil(Int, T_FINAL/tf)
t_span = (0., tf)
abstol = 100

# define the system
global mass_system = MassSystem(N_INITIAL,velocity)
mass_system.velocities .-= mean(mass_system.velocities, dims=1)
mass_system.root = build_tree!(mass_system)

# start simulation
n=2
t = @elapsed all_positions, energy, lost_masses_energy = start_cal(n, mass_system, t_span, abstol)
n = size(energy)[1]
# #%%
# # make a folder for saving the results
# gr()
# global folder = create_folder()
# new_folder_rename = folder[1:end-1] * format("_eps_{}_N_mass_{}_repeat_ode_{}_v_{}_time_{:.2f}_m_jullia/",abstol,N_INITIAL,n,velocity,t/60)
# mv(folder, new_folder_rename)
# folder = new_folder_rename
#
# # save results to files
# save(folder, mass_system, [energy, lost_masses_energy], ["energy", "lost_masses_energy"])
# save_figures(1, all_positions, folder)
# # py"gif_"(folder,"animated")

#%%
# plot the graphs
# plotly()
time_series = (0:n-1) .* tf
total_energy1 = sum(energy, dims=2)
total_energy2 = sum(energy.-lost_masses_energy, dims=2)
# Plots.plot(time_series, -1 .+ total_energy1 ./ total_energy1[1], label=L"Energy\ Conservation")
# Plots.plot(time_series, total_energy1 , label=L"Energy\ Conservation")
# Plots.plot!(time_series, -1 .+ total_energy2 ./ total_energy2[1], label=L"Energy\ Conservation\ with\ lost")

# Plots.plot(time_series, energy[:,1], label="Kinetic")
# Plots.plot!(time_series, energy[:,2], label="potintional")
# Plots.plot!(time_series, (energy.-lost_masses_energy)[:,1], label="Kinetic with lost")
# Plots.plot!(time_series, (energy.-lost_masses_energy)[:,2], label="potintional with lost")

# Plots.savefig(folder*"energy.png")
# Plots.plot(time_series, -2 .* energy[:,1] ./ energy[:,2],label=L"\dfrac{E_k}{E_p}")
# Plots.plot!(time_series, -2 .* (energy[:,1] .- lost_masses_energy[:,1]) ./ (energy[:,2] .- lost_masses_energy[:,2]),label=L"\dfrac{E_k}{E_p}+lost\ masses")
# Plots.savefig(folder*"ratio.png")
# xlabel!("t"*" [astrnumical units]")

# plot((0:length(all_positions)-1) .* tf,R2.([p[1,:]-p[2,:] for p in all_positions]))
# find_density(mass_system)
# Plots.savefig(folder*"king.png")
#%%
# mass_system
# end


# println("finished to calculate system start on average")
# t_span = (0., tf*3)
# n = 50

# all_positions, energy, lost_masses = start_cal(n, mass_system, t_span, abstol)
#
# radii = []
# for pos in all_positions
#     COM = mean(pos, dims=1)[1,:]
#     push!(radii, sqrt.((pos[:,1] .- COM[1]).^2 .+(pos[:,2] .- COM[2]).^2 .+(pos[:,3] .- COM[3]).^2) )
# end
# radii2 = vcat(radii...)
# fold = string(folder,"MassSystem/")
# open(fold*"radii2.txt", "w") do io
#     write(io, JSON.json(radii2))
# end
# find_density(radii2)
# find_density(mass_system)
