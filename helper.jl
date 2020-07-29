# MAX_BOX_SIZE = 666e3  # Parsec

function scatter_(mass_system::MassSystem, cam=(40,50))
    p = mass_system.positions
    scatter_(p, cam)
end

function scatter_(p::Array{Float64,2}, cam=(40,50))
    Plots.scatter(p[:,1],p[:,2],p[:,3], camera = cam)
    xlm = ylm = zlm = (-MAX_BOX_SIZE/30, MAX_BOX_SIZE/30)  # graph to reproduce the magnification from mousing
    xlims!(xlm[1], xlm[2])  # Reproduce magnification
    ylims!(ylm[1], ylm[2])
    zlims!(zlm[1], zlm[2])
end
##


macro Name(arg)
    string(arg)
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
function load(folder, args...)
    system = MassSystem(1,1.)

    fold = string(folder,"MassSystem/")
	p = JSON.parse(open(f->read(f, String), fold*"positions.txt"));
    v = JSON.parse(open(f->read(f, String), fold*"velocities.txt"));
    m = JSON.parse(open(f->read(f, String), fold*"eachmass.txt"));
    args_out = []
    for var_name in args
        var = JSON.parse(open(f->read(f, String), fold*var_name*".txt"))
        push!(args_out, var)
    end


    system.N = size(p[1])[1]
    system.each_mass = m

    system.positions = hcat(p...)
    system.velocities = hcat(v...)
    system.forces = zeros(system.N, 3)

    system.root = build_tree!(system)
    system, args_out...
end

function save_figures(num, save_positions, folder)
    i=0
    anim = @animate for p in save_positions
        scatter_(p)
        Plots.savefig(string(folder,i,".png"))
        i = i + 1
    end;
    gif(anim, folder*"gif.gif",fps=5);
end

function show_gif(save_positions,cam)
    anim = @animate for p in save_positions
        scatter_(p,cam)
    end;
	gif(anim, fps=5)
end

function create_folder()
    cd("/home/elyco/github/End_Project/")
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
