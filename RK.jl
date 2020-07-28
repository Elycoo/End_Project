function make_solver(f::Function, x_range::Tuple{Float64,Float64}, method::String, eps::Float64,p::MassSystem)
    function ode(y_0::Array{Float64,2})
        dim = length(y_0)
        y = zeros(dim,1)
        y = y_0
        x = [x_range[1]]

        h = eps
        delta::Float64 = eps * h / (x_range[2] - x_range[1])
        eps_new::Float64 = eps

        i = 0
        y1 = similar(y_0[:,1])
        y2_tmp = similar(y_0[:,1])
        y2 = similar(y_0[:,1])
        while x[end] < x_range[2]
            i = i + 1
            while true
                y1[:] = solve_small_step(f, x[i], y[:, i], h, method,p)
                y2_tmp[:] = solve_small_step(f, x[i], y[:, i], h/2, method,p)
                y2[:] = solve_small_step(f, x[i]+h/2, y2_tmp, h/2, method,p)

                dist_between_y1_y2::Float64 = maximum(abs.(y1 - y2))
                if dist_between_y1_y2 < delta
                    x = vcat(x, [x[end]+h])

                    if dist_between_y1_y2 > delta/2
                        # println("adjust h->0.9h")
                        h = h*0.9
                    elseif dist_between_y1_y2 > delta / 4
                        # println("h is good")
                    else
                        # println("adjust h->1.1h")
                        h = h*1.1
                    end

                    eps_new = eps_new - dist_between_y1_y2 / 2
                    delta = eps_new*h/(x_range[2]-x[i])
                    break
                else
                    # println("h is big! adjust h->0.5h")
                    h = h / 2
                end
            end
            # Joining the two calculation together
            # By completing the formula from the course note for h^4 we are joining now
            # the two calculations that we did.
            y_n = zeros(dim, 1)
            if method == "RK3"
                y_n[:,1] = 4 * y2 / 3 - y1 / 3
            elseif method == "RK4"
                y_n[:,1] = 8 * y2 / 7 - y1 / 7
            else
                y_n[:,1] = y2
            end

            y = hcat(y, y_n)
        end
        if x[end] < x_range[2]
            println("Warning, not reach to the end!")
        end
        return x, y
    end
    return ode
end

function solve_small_step(f::Function, a::Float64, f_a::Array{Float64,1}, h::Float64, method::String,p::MassSystem)::Array{Float64,1}
    # if method == 1
    #     y_new = f_a + h * f(a, f_a,p)
    # elseif method == 2
    #     k = h * f(a, f_a,p)
    #     y_new = f_a + h * f(a + h / 2, f_a + k / 2,p)
    if method == "RK3"
        k1 = h * f(a, f_a,p)
        k2 = h * f(a + h / 2, f_a + k1 / 2,p)
        k3 = h * f(a + h, f_a + 2 * k2 - k1,p)
        y_new = f_a + (k1 + 4 * k2 + k3) / 6
    elseif method == "RK4"
        k1 = h * f(a, f_a,p)
        k2 = h * f(a + h / 2, f_a + k1 / 2,p)
        k3 = h * f(a + h / 2, f_a + k2 / 2,p)
        k4 = h * f(a + h, f_a + k3,p)
        y_new = f_a + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    end
    y_new
end

#
# using Plots
# function check_ode()
# f(x, y, p) = [y[2],-y[1]]
#
# x0 = 0.
# xf = 10.
# y0 = zeros(2, 1)
# y0[:, 1] = [1., 0.]
# eps = 1e-3
# ode_solver = make_solver(f, (x0, xf), "RK4", eps)
#
# x, y = ode_solver(y0)
# y_exact = vcat(cos.(x), -sin.(x))
#
#
# gr()
# plot(x[1,:],y',label="RK.jl",marker = (:circle, 4, .6))
# plot!(x[1,:],y_exact',label="exact")
#
# end
# check_ode()
# using ..test
