import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from numba import jit

plt.rcParams["font.size"] = "16"


def make_solver(f, x_range, method, eps):
    #@jit
    def ode(y_0):
        dim = len(y_0)
        y = np.zeros((dim, 1))
        x = np.array([0.])
        y = y_0
        x[0] = x_range[0]

        h = eps
        delta = eps*h/(x_range[1] - x_range[0])
        eps_new = eps

        i = -1
        while x[-1] < x_range[1]:
            i = i + 1
            while True:
                y1 = solve_small_step(f, x[i], y[:, i], h, method)
                y2_tmp = solve_small_step(f, x[i], y[:, i], h/2, method)
                y2 = solve_small_step(f, x[i]+h/2, y2_tmp, h/2, method)

                dist_between_y1_y2 = np.abs(y1 - y2).max()
                if dist_between_y1_y2 < delta:
                    x = np.hstack((x, np.array([x[-1]+h])))

                    if dist_between_y1_y2 > delta/2:
                        # print('adjust h->0.9h')
                        h = h*0.9
                    elif dist_between_y1_y2 > delta / 4:
                        # print('h is good')
                        pass
                    else:
                        # print('adjust h->1.1h')
                        h = h*1.1

                    eps_new = eps_new - dist_between_y1_y2/2
                    delta = eps_new*h/(x_range[1]-x[i])
                    break
                else:
                    # print('h is big! adjust h->0.5h')
                    h = h / 2

            # Joining the two calculation together
            # By completing the formula from the course note for h^4 we are joining now
            # the two calculations that we did.
            y_n = np.zeros((dim, 1))
            if method == 'RK3':
                y_n[:,0] = 4 * y2 / 3 - y1 / 3
            elif method == 'RK4':
                y_n[:,0] = 8 * y2 / 7 - y1 / 7
            else:
                y_n[:,0] = y2

            y = np.hstack((y, y_n))
        if x[-1] < x_range[1]:
            print('Warning, not reach to the end!')
        # print()
        # return np.vstack((x.reshape((-1,1)), y))
        return x, y
    return ode


#@jit
def solve_small_step(f, a, f_a, h, method):
    if method == 1:
        y_new = f_a + h * f(a, f_a)
    elif method == 2:
        k = h * f(a, f_a)
        y_new = f_a + h * f(a + h / 2, f_a + k / 2)
    elif method == 'RK3':
        k1 = h * f(a, f_a)
        k2 = h * f(a + h / 2, f_a + k1 / 2)
        k3 = h * f(a + h, f_a + 2 * k2 - k1)
        y_new = f_a + (k1 + 4 * k2 + k3) / 6
    elif method == 'RK4':
        k1 = h * f(a, f_a)
        k2 = h * f(a + h / 2, f_a + k1 / 2)
        k3 = h * f(a + h / 2, f_a + k2 / 2)
        k4 = h * f(a + h, f_a + k3)
        y_new = f_a + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y_new


def check_ode():
    # def f(x,y):
        # y = y_0 * np.exp(-2*x_ans)
        # return -2*y

    def f(x,y):
        # y = y_0 * np.exp(-x ** 2 / 2)
        return -x*y

    y_0 = np.array([3.])
    x_0 = 0
    x_end = 10
    eps = 1e-3

    x_RK3, y_ans_RK3 = ode(f, (x_0, x_end), y_0, 'RK3', eps)
    x_RK4, y_ans_RK4 = ode(f, (x_0, x_end), y_0, 'RK4', eps)
    sol = solve_ivp(f, (x_0, x_end), y_0)

    # The exact solution
    # y = lambda x: y_0 * np.exp(-2 * x)
    y = lambda x: y_0* np.exp(-x**2/2)

    plt.figure(1)
    plt.plot(x_RK4, y(x_RK4), label='analytical results', linewidth=2)
    plt.plot(x_RK4, y_ans_RK4[0,:],'o', label='Runga-Kuta order 4', linewidth=1)
    plt.legend()
    plt.show()

    plt.figure(1)
    plt.plot(x_RK4, abs(y_ans_RK4[0,:] - y(x_RK4)), label='Absolute error - My', linewidth=2)
    # plt.plot(sol.t, abs(sol.y[0,:] - y(sol.t)), label='Absolute error - Scipy', linewidth=2)
    plt.yscale('log')
    plt.legend()
    plt.show()


def check_system_ode():
    #@jit
    def f(x, y_array):
        return np.array([y_array[1],
                         -y_array[0]])

    x0 = 1.
    xf = 100.
    y0 = np.zeros((2, 1))
    y0[:, 0] = np.array([1., 0.])
    eps = 1e-3
    ode_solver = make_solver(f, (x0, xf), 'RK4', eps)
    tic.tic()
    x, y = ode_solver(y0)
    tic.toc(restart=True)
    x, y = ode_solver(y0)
    tic.toc()
    y_exact = np.array([np.cos(x), -np.sin(x)])

    plt.figure()
    plt.plot(x, y.T, 'o')
    plt.plot(x, y.T)
    plt.legend(['$y_1$', '$y_2$', '$\cos$', '$-\sin$'])
    plt.show()

    plt.figure()
    plt.plot(x, np.abs(y.T/y_exact.T - 1))
    plt.legend(['Relative Error'])
    plt.yscale('log')
    plt.show()


def check_steps_calls():
    global count

    a = 0.1
    b = np.array([-5., -1., 1., 4., 7.])
    count = 0

    def f(x, y):
        global count
        count += 1
        dydx = 1/(a*np.sqrt(np.pi)) * np.sum( np.exp( -(x-b)**2/a**2 ) )
        return dydx

    y0 = 0
    x_range = (-10, 10)

    print('%15s %15s %15s %25s %15s' % ('RK order', 'Requsted accuracy', 'No. of steps', 'No. of function calls', 'Relative error'))
    for method in ['RK3', 'RK4']:
        for eps in [1e-3, 1e-5, 1e-7, 1e-9]:
            count = 0
            y_final = ode(f, x_range, [y0], method, eps)[1]
            print('%15s %15g %15d %25d %15.4g' % (method[-1], eps, len(y_final[0]), count, np.abs(y_final[0,-1]/5-1)))


if __name__ == '__main__':
    # Part 1
    # check_ode()
    from pytictoc import TicToc
    tic = TicToc()
    check_system_ode()

