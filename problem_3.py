import qmparticle
import numpy as np
from math import ceil
import matplotlib.pyplot as plt
import tools
import time
import pickle as pkl

t0 = time.time()

L = 20
N = 2000
k0 = 20
T = 0.5

barrier_height = k0**2 / 4
barrier_width = L / 50

space, step_size = np.linspace(0, L, N, endpoint=True, retstep=True)

sigXs = [0.5, 1.0, 1.5, 2.0]


def potential(x):
    pot = []
    for xs in x:
        if (L / 2 - barrier_width / 2) < xs < (L / 2 + barrier_width / 2):
            pot.append(barrier_height)
        else:
            pot.append(0)
    return np.array(pot)


def simulate():
    for sigX in sigXs:
        print("Calculating for sigX={0}".format(sigX))

        gaussian = tools.create_gaussian(space, 5, sigX, k0, step_size)
        qmp = qmparticle.QMParticle(space, lambda x: gaussian, potential)

        iterations = int(T / qmp.dt)

        animation_time = 10
        fps = 60
        frames = animation_time * fps

        Psi_real = [gaussian.real]
        Psi_imag = [gaussian.imag]
        rho_sq = [gaussian.real ** 2 + gaussian.imag ** 2]

        for i in range(frames):
            for j in range(iterations // frames):
                qmp.step()
            Psi_real.append(qmp.prev_Psi.real)
            Psi_imag.append(qmp.prev_Psi.imag)
            rho_sq.append(qmp.prev_Psi.real ** 2 + qmp.prev_Psi.imag ** 2)
            print(i)

        pkl.dump([Psi_real, Psi_imag, rho_sq], open("problem3_sigX={0}.p".format(sigX), "wb"))


def display(sigX):
    values = pkl.load(open("problem3_sigX={0}.p".format(sigX), 'rb'))

    fig1, ax1 = plt.subplots(num=1)
    ax1.set_title(r"$\Psi$ at $x=5$ for $\sigma_x={0}$".format(sigX))
    ax1.set_xlabel(r"$x$", size=18)
    ax1.set_ylabel(r"$|\Psi|^2$", size=18)
    ax1.plot(space, values[0][0], label=r"$\Psi_R$")
    ax1.plot(space, values[1][0], label=r"$\Psi_I$")
    ax1.legend()
    ax1.grid(True)

    fig2, ax2 = plt.subplots(num=2)
    ax2.set_title(r"$\Psi$ at $x=15$ for $\sigma_x={0}$".format(sigX))
    ax2.set_xlabel(r"$x$", size=18)
    ax2.set_ylabel(r"$|\Psi|^2$", size=18)
    ax2.plot(space, values[0][-1], label=r"$\Psi_R$")
    ax2.plot(space, values[1][-1], label=r"$\Psi_I$")
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    ax2.legend()
    ax2.grid(True)

    plt.show()


for sigX in sigXs:
    display(sigX)


t1 = time.time()
print("Total time: {0} s".format(t1 - t0))
