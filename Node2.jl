module name
export Node
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

end  # module
