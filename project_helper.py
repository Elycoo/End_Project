from matplotlib import pyplot as plt

# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D  # <--- This is important for 3d plotting

# import matplotlib
# matplotlib.use('Qt5Agg')

import os
from datetime import datetime

plt.rcParams["font.size"] = "16"
MAX_BOX_SIZE = 666e3


def save_figures(num, save_positions, folder, show=False, save=True):
    fig = plt.figure(num)
    ax = fig.add_subplot(111, projection='3d')
    for i, state in enumerate(save_positions):
        if i == 0:
            scat = ax.scatter3D(*state.T)
            xlm = ylm = zlm = (-MAX_BOX_SIZE/3, MAX_BOX_SIZE/3)  # graph to reproduce the magnification from mousing
            ax.set_xlim3d(xlm[0], xlm[1])  # Reproduce magnification
            ax.set_ylim3d(ylm[0], ylm[1])  # ...
            ax.set_zlim3d(zlm[0], zlm[1])  #

            # azm = ax.azim
            # ele = ax.elev
            # ax.view_init(elev=ele, azim=azm)  # Reproduce view
        else:
            scat.remove()
            scat = ax.scatter3D(*state.T, c='C0')
        if show:
            plt.show()
            # fig.canvas.draw()
            # fig.canvas.flush_events()
        if save:
            plt.savefig(folder + f'{i}.png', bbox_inches='tight', dpi=100)


def gif(folder, name):
    """
    Create a gif from png images in folder and save them as 'folder/name'
    """
    import imageio
    import os
    import re

    filename = [fn for fn in os.listdir(folder) if fn.endswith('.png')]
    filename.sort(key=lambda f: int(re.sub('\D', '', f)))
    images = [imageio.imread(folder + fn) for fn in filename]

    imageio.mimsave(folder + name + '.gif', images)


def beep():
    # Make a beep sound
    duration = 100  # milliseconds
    freq = 440  # Hz
    if os.name != 'posix':
        import winsound
        winsound.Beep(freq, duration)
    else:
        pass
        # os.system('play -nq -t alsa synth {} sine {}'.format(duration//1000, freq))


def create_folder():
    day = datetime.now().strftime('%m_%d')
    hour = datetime.now().strftime('%H-%M')
    if not os.path.exists(f'./results/{day}'):
        os.mkdir(f'./results/{day}/')
    if not os.path.exists(f'./results/{day}/{hour}/'):
        os.mkdir(f'./results/{day}/{hour}/')
    return f'./results/{day}/{hour}/'
