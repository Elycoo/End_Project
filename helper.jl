function scatter_(mass_system::MassSystem, cam=(40,50))
    p = mass_system.positions
    scatter_(p, cam)
end

function scatter_(p::Array{Float64,2}, cam=(40,50))
    Plots.scatter(p[:,1],p[:,2],p[:,3], camera = cam)
    xlm = ylm = zlm = (-MAX_BOX_SIZE/3, MAX_BOX_SIZE/3)  # graph to reproduce the magnification from mousing
    xlims!(xlm[1], xlm[2])  # Reproduce magnification
    ylims!(ylm[1], ylm[2])
    zlims!(zlm[1], zlm[2])
end


function save(folder, system, args...)
    fold = string(folder,"MassSystem/")
    mkdir(fold)
    open(fold*"positions.txt", "w") do io
        write(io, JSON.json(system.positions))
    end
    open(fold*"velocities.txt", "w") do io
        write(io, JSON.json(system.velocities))
    end
    open(fold*"eachmass.txt", "w") do io
        write(io, JSON.json(system.each_mass))
    end
    for i in 1:length(args[1])
        name_var = args[2][i]
        open(fold*name_var*".txt", "w") do io
            write(io, JSON.json(args[1][i]))
        end
    end
end
"""
	load(folder, args...)

# Use it as follow:
```julia-repl
julia> folder = "/path/to/some/folder"; # that contain the relevant MassSystem data
julia> mass_system, sim_data, enr, los_enr = BranesHut.load(folder,"energy","lost_masses_energy");
```

"""
function load(folder, args...)
    system = MassSystem(1,1.)

    fold = string(folder,"MassSystem/")
	p = JSON.parse(open(f->read(f, String), fold*"positions.txt"));
    v = JSON.parse(open(f->read(f, String), fold*"velocities.txt"));
    m = JSON.parse(open(f->read(f, String), fold*"eachmass.txt"));
    args_out = []
    for var_name in args
        var = JSON.parse(open(f->read(f, String), fold*var_name*".txt"))
        push!(args_out, hcat(var...))
    end


    system.N = size(p[1])[1]
    system.each_mass = m

    system.positions = hcat(p...)
    system.velocities = hcat(v...)
    system.forces = zeros(system.N, 3)

    system.root = build_tree!(system)
	sim_data = SimData((0.,80.), length(args_out[1][:,1]), 100., folder)
    system, sim_data, args_out...
end

function save_figures(num, save_positions, folder)
    i=0
    anim = @animate for p in save_positions
        scatter_(p)
        Plots.savefig(string(folder,i,".png"))
        i = i + 1
    end;
    gif(anim, folder*"gif.gif",fps=5)
end

function show_gif(save_positions,cam)
    anim = @animate for p in save_positions
        scatter_(p,cam)
    end;
	gif(anim, fps=5)
end

function create_folder()
	if !Sys.iswindows()
    	cd("/home/elyco/github/End_Project/")
	end
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
    distance_vec = node.center_of_mass - point  # attractive energy
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

# using Peaks
function check_kepler_third_law()
	ratio = zeros(1)
	for i in 1:length(ratio)
	    seed!(1243333333)
		# prepare system
	    velocity = 15.6
	    mass_system = MassSystem(N_INITIAL,velocity)
	    mass_system.positions = rand()*[0 0 -R_INITIAL;0 0 R_INITIAL]*20
	    mass_system.velocities = [0 velocity 0;0 -velocity 0]
	    mass_system.velocities .-= mean(mass_system.velocities, dims=1)
	    mass_system.root = build_tree!(mass_system)

	    # start simulation
		dof = mass_system.N*6
	    tf = 400000.
	    t_span = (0.,tf)
	    abstol = .01

	    y_0 = zeros(mass_system.N*6,1)
	    y_0[:] = get_current_values(mass_system)
	    ode_solver = make_solver(ode_to_solve_my_RK, t_span, "RK4", abstol, mass_system)
	    time_series, y = ode_solver(y_0)
		y = [reshape(a[:],mass_system.N, 3) for a in eachcol(y[1:Int(dof//2),:])]

		# analyze results
	    r = sqrt.(R2.([p[1,:]-p[2,:] for p in y]))
	    peaks = maxima(r)
	    mins = minima(r)
	    a = (r[peaks[1]] + r[mins[1]])/2
	    T = mean([time_series[peaks[1]] - time_series[peaks[2]] ; time_series[mins[1]] - time_series[mins[2]]])

		# compare to knonw value
	    rat = GRAVITATIONAL_CONSTANT*M_TOT*T^2/a/(a+SOFT_PARAM)^2/4/pi^2
	    no_correction = GRAVITATIONAL_CONSTANT*M_TOT*T^2/a^3/4/pi^2
	    ratio[i] = rat
		@show no_correction

		#  plot and animate
		plo = plot(time_series, r, legend=:bottomright)
	    scatter!(time_series[peaks], r[peaks])
	    scatter!(time_series[mins], r[mins])
	    inds = floor.(Int,collect(range(1,stop=length(y),length=100)))
	    anim = show_gif(y[inds],(40,70))
	    display(anim)
	    display(plo)
	end
	@show ratio

#=
results for these configuration:
seed!(1243333333)
mass_system.positions = rand()*[0 0 -R_INITIAL;0 0 R_INITIAL]*20
velocity = 15.6
tf = 400000.
abstol = .01

no_correction = 0.9996824314578093
ratio = [0.9948236845203343]
=#
end


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
end # function get_more_data_to_average
