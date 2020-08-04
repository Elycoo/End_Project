@everywhere module BranesHut
#%%
#=
project_jul:
- Julia version: 1.0.4
- Author: elyco
- Date: 2020-07-19
=#

### import and constants ###
using Distributed
using LinearAlgebra: norm
using Distributions: mean
using Random: seed!
using KernelDensity: kde, pdf
using LsqFit: curve_fit

# using Debugger
using Plots
using LaTeXStrings
using Dates
using JSON: JSON, parse
using Formatting: format

include("StructModule.jl")
using ..StructModule
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
const N_INITIAL = 5000 # The total number of star in the simulation

const M_TOT                   = 1e11  # [Sun masses]  The total mass of all the stars
const R_INITIAL               = 50e3  # [Parsec]  The Radius that all the masses start inside
const SOFT_PARAM              = 1e3  # [Parsec]
const MAX_BOX_SIZE            = 666e3  # [Parsec]
# const T_FINAL                 = 20e9  # [years], = 20453 in units time of the exercise
const T_FINAL                 = 20453  # in units time of the exercise
const THETA                   = 1      # parameter that control how deep we need the gey inside the recursion
const GRAVITATIONAL_CONSTANT  = 4.30091e-3  # in the units of the exercise

#%% Trees construct
"""
    build_tree!(system::MassSystem)

Build the tree that contain all the information of the system in OctTree Structure
Return the root of the tree which is attribute of MassSystem
"""
function build_tree!(system::MassSystem)
    borders = get_current_box_size(system)  # x_lim, y_lim, z_lim

    # define a new node with all the masses
    root = Node(borders)
    root.masses_indices = collect(1:system.N)
    root.center_of_mass = mean(system.positions, dims=1)[1,:]
    root.mass_count = system.N

    build_tree_helper!(system, root)
    return root
end

function build_tree_helper!(system::MassSystem, node::Node)
    # divide the tree into 8 parts
    xlim = node.borders[:,1]
    ylim = node.borders[:,2]
    zlim = node.borders[:,3]
    middle_limits = [mean(lim,dims=1) for lim in (xlim,ylim,zlim)]

    borders8 = [[sort!([xlim[i], middle_limits[1][1]]),
                 sort!([ylim[j], middle_limits[2][1]]),
                 sort!([zlim[k], middle_limits[3][1]])]
                for i in 1:2 for j in 1:2 for k in 1:2]

    borders8 = [hcat(b[1],b[2],b[3]) for b in borders8]

    # for each part check its properties and if neede recursive build its children
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
""""
    fill_attributes!(system, node)

Save to node all the relevant infromation about that node
"""
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
    calculate_force!(system)

Initiate the calculation of the force for each point mass we are saving the force that act on it.
there are two options for the calculation, depend on I want to parralelize it or not.
"""
function calculate_force!(system)
    # =
    tmp = convert(SharedArray{Float64,2},zeros(system.N, 3))
    @sync begin
        # pmap(i-> (
        #     point = system.positions[i,:];
        #     tmp[i,:] = calculate_force_helper!(system, system.root, point)
        #     ), 1:system.N)
        @distributed for i in 1:system.N
            point = system.positions[i,:]
            tmp[i,:] .= calculate_force_helper!(system, system.root, point)
        end
    end
    system.forces = tmp
    #= =# #=
    system.forces = zeros(system.N, 3)
    i = 1
    for point in eachrow(system.positions)
        system.forces[i,:] = calculate_force_helper!(system, system.root, point)
        i = i + 1
    end
    # =#
end
"""
    calculate_force_helper!(system, node, point)
Recursive function return the force acting on `point` from all the masses in `node`
"""

@everywhere function calculate_force_helper!(system, node, point)
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
        if distance < node.diagonal * THETA || point_in_box(point, node.borders)
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
"""
Check if some of the mass escape from the box of calculation and if so erase them from the simulation
"""
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

R2(x::Float64,y::Float64,z::Float64) = x^2+y^2+z^2
R2(r::Array{Float64,1}) = r[1]^2+r[2]^2+r[3]^2
R2(r::Array{Float64,2}) = @. r[:,1]^2+r[:,2]^2+r[:,3]^2
function calculate_energy(mass_system::MassSystem)
    potential_energy = calculate_potential_energy(mass_system)/2    # no double counting
    v_squre = R2(mass_system.velocities)
    kinectic_energy = 0.5 * mass_system.each_mass * sum(v_squre)
    [kinectic_energy, potential_energy]
end

#%%
"""
    ode_to_solve_my_RK(t::Float64,y::Array{Float64,1},system::MassSystem)
Function that translate the RK way that it calcuate thing and the way I implement the problem here.
"""
function ode_to_solve_my_RK(t::Float64,y::Array{Float64,1},system::MassSystem)
    dof = system.N * 3
    set_values_from_flat!(y,system)
    system.root = build_tree!(system)
    calculate_force!(system)

    drdt = y[dof+1:end]
    dvdt = system.forces[:] ./ system.each_mass
    dy = vcat(drdt, dvdt)
end


function start_cal(mass_system::MassSystem, sim_data)
    count_dead = 0
    save_positions = []
    energy = []
    energy_of_lost_massess = []
    kinetic_energy_of_lost_massess = []
    for i in 1:sim_data.n
        println(i)  # follow the progress of the simulation
        y_0 = zeros(mass_system.N*6, 1)
        y_0[:] = get_current_values(mass_system)
        if mass_system.N < N_INITIAL - count_dead
            count_dead = N_INITIAL - mass_system.N
            println(count_dead, " particles are deads")
        end

        ode_solver = make_solver(ode_to_solve_my_RK, sim_data.t_span, "RK4", sim_data.abstol, mass_system)
        sol = ode_solver(y_0)[2][:,end]  # take the last state from the RK
        set_values_from_flat!(sol, mass_system)

        # calcuate the energy difference before and after we remove the exceeded masses
        energy_before = calculate_energy(mass_system)
        remove_exceeds_masses(mass_system)
        energy_after = calculate_energy(mass_system)
        energy_lost = energy_after .- energy_before

        # save the energy
        push!(energy, energy_before)
        push!(energy_of_lost_massess, energy_lost )

        if i % 2 == 0
            # save it at not every step
            push!(save_positions,mass_system.positions)
        end
        if mass_system.N == 0
            println("Simulation is over, no more particles")
            return save_positions, vcat(energy'...) , vcat(cumsum(energy_of_lost_massess)'...)
        end
    end

    save_positions, vcat(energy'...) , vcat(cumsum(energy_of_lost_massess)'...)
end
#%%
function run(;velocity = 80., save_data=true)
    seed!(421364) # seed the random number generator for reproducibility

    # define parameters of calculation
    tf = 80.
    t_span = (0., tf)
    n = ceil(Int, T_FINAL/tf)
    abstol = 100.
    sim_data = SimData(t_span, n, abstol)  # save the simulation data in a compact object

    # initialize the system
    mass_system = MassSystem(N_INITIAL,velocity)
    @show N_INITIAL velocity abstol
    mass_system.velocities .-= mean(mass_system.velocities, dims=1) # shift to center of mass frame
    mass_system.root = build_tree!(mass_system)

    # start simulation
    t = @elapsed all_positions, energy, lost_masses_energy = start_cal(mass_system, sim_data)
    n = size(energy)[1]
    #%%
    # return mass_system, sim_data, all_positions, energy, lost_masses_energy
    if save_data || (println("No saving to file"); false)
        println("saving data to folder...");
        # make a folder for saving the results
        gr() # use gr() backend (faster and simpler then plotly)
        folder = create_folder()
        new_folder_rename = folder[1:end-1] * format("_eps_{}_N_mass_{}_repeat_ode_{}_v_{}_time_{:.2f}_m_THETA_{}_jullia/",abstol,N_INITIAL,n,velocity,t/60,THETA)
        mv(folder, new_folder_rename)
        folder = new_folder_rename
        sim_data.folder = folder

        # save results to files
        save_figures(1, all_positions, folder)
        save(folder, mass_system, [energy, lost_masses_energy], ["energy", "lost_masses_energy"])
        # show_gif(all_positions, (40,70))
    end
    return mass_system, sim_data, all_positions, energy, lost_masses_energy
end # run

#%%
function run_energy_graphs(sim_data, all_positions, energy, lost_masses_energy; save_data=false)
    # plot the graphs
    # plotly()
    tf = sim_data.t_span[2]
    folder = sim_data.folder

    p1 = plot(title="Total Energy")
    p2 = plot(title="Kinetic Energy")
    p3 = plot(title="Potential Energy")
    p4 = plot(title="Rel error of Energy Conservation")
    p5 = plot(title="Virial Thm. relation")
    p6 = plot(title="energy")

    time_series = (0:sim_data.n-1) .* tf
    total_energy1 = sum(energy, dims=2)
    total_energy2 = sum(energy.-lost_masses_energy, dims=2)
    plot!(p1,time_series, total_energy1 , label=L"Energy\ Conservation")
    plot!(p1,time_series, total_energy2, label=L"Energy\ Conservation\ with\ lost")

    plot!(p2,time_series, energy[:,1], label="Kinetic")
    plot!(p2,time_series, (energy.-lost_masses_energy)[:,1], label="Kinetic with lost")

    plot!(p3,time_series, energy[:,2], label="Potential")
    plot!(p3,time_series, (energy.-lost_masses_energy)[:,2], label="Potential with lost")

    plot!(p4, time_series, -1 .+ total_energy1 ./ total_energy1[1], label=L"Energy\ Conservation")
    plot!(p4, time_series, -1 .+ total_energy2 ./ total_energy2[1], label=L"Energy\ Conservation\ with\ lost")


    save_data && savefig(folder*"energy.png")
    plot!(p5,time_series, -2 .* energy[:,1] ./ energy[:,2],label=L"\dfrac{2E_k}{E_p}")
    plot!(p5,time_series, -2 .* (energy[:,1] .- lost_masses_energy[:,1]) ./ (energy[:,2] .- lost_masses_energy[:,2]),label=L"\dfrac{2E_k}{E_p}+lost\ masses")
    save_data && savefig(folder*"ratio.png")
    # xlabel!("t"*" [astrnumical units]")

    # length(all_p[1])==6 && plot!(p6,(0:length(all_positions)-1) .* tf,R2.([p[1,:]-p[2,:] for p in all_positions]))
    plot(p1,p4,p5,p2,p3,legend=false,size=(600,1600), layout=(5,1))
end # run_energy_graphs

#%%
function find_density(system::MassSystem; save_data=false, folder="")
    p = system.positions
    COM = system.root.center_of_mass
    radii = @. sqrt((p[:,1] - COM[1])^2 +(p[:,2] - COM[2])^2 +(p[:,2] - COM[3])^2)
    find_density(radii, save_data=false, folder=folder)
end
function find_density(radii::Array{Float64,1}; save_data=false, folder="")
    gr()
    xdata = range(minimum(radii), maximum(radii),length=10^3)
    xdata = range(minimum(radii), 10. ^5.5,length=5*10^2)
    p = kde(radii)
    ydata = pdf(p,xdata)
    # inds = [i for i in 1:length(ydata) if ydata[i]>0 && xdata[i]<10^5]
    # xdata = xdata[inds]
    # ydata = ydata[inds]


    # curve_fit
    @. model(x, p) =  p[1]*x^2/(1 + (x/p[2])^p[3] )^(p[4])
    p0 = [1e-5,10^4.7,5.13,2.2]
    p0 = [1e-14, 4e4, 2.1, 3.05]
    p0=[6.52e-10, 7.1e+04, 1., 7.]
    p0 = [1.e-13, 159040., 1., 14.] #[1.102829573732584e-13, 159040.00000000675, 1.0015000953676354, 13.999914095981763]
    p0 = [1.e-13, 159040., 2., 2.5] #[9.670895322014411e-14, 75887.91935011698, 1.4498570607553147, 6.835788293565527]
    p0 = [1.e-13, 159040., 2., 2.5]

    lb = [0 ,0.,0.,0.]  # lower bound
    # ub = [1 ,1e6,3.,3.]  # upper bound
    ub = [Inf ,Inf,Inf,Inf]  # upper bound
    fit = curve_fit(model, xdata, ydata, p0 ,lower=lb,upper=ub)
    println(fit.param)
    println(sum(fit.resid.^2))

    plo = plot(xdata, ydata, xaxis=:log,yaxis=:log, marker="o", label="simulation")
    # plo = plot(xdata,ydata, marker="o", label="simulation")
    plot!(xdata, model(xdata,fit.param), label="fit")
    save_data && savefig(folder*"king"*string(fit.param...)*".png")
    plo
    # density
    # fit
end
#%%

function get_more_data_to_average()
    println("finished to calculate system start on average")
    t_span = (0., tf*3)
    n = 50

    all_positions, energy, lost_masses = start_cal(mass_system, sim_data)

    radii = []
    for pos in all_positions
        COM = mean(pos, dims=1)[1,:]
        push!(radii, sqrt.((pos[:,1] .- COM[1]).^2 .+(pos[:,2] .- COM[2]).^2 .+(pos[:,3] .- COM[3]).^2) )
    end
    radii2 = vcat(radii...)
    fold = string(folder,"MassSystem/")
    open(fold*"radii2.txt", "w") do io
        write(io, JSON.json(radii2))
    end
    find_density(radii2)
    find_density(mass_system)
end # function check_kepler_third_law

using Main: calculate_force_helper!,point_in_box

end  # module BranesHut

function run_all()
    for v in [65., 80., 95.]
        mass_system, sim_data, all_p, erg, lost_erg  = BranesHut.run(velocity=v, save_data=true)
        BranesHut.run_energy_graphs(sim_data, all_p, erg, lost_erg, save_data=true)
        BranesHut.find_density(mass_system, save_data=true, folder=sim_data.folder)
    end
end
