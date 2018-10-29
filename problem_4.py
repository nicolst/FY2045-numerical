import qmparticle
import numpy as np
from progress.bar import Bar
import matplotlib.pyplot as plt
import tools
import time
import pickle as pkl
import threading

t0 = time.time()

L = 20
N = 2000
k0 = 20
T = 0.5
sigX = 1.0
l = L / 50

space, step_size = np.linspace(0, L, N, endpoint=True, retstep=True)


def potential(xs, barrier_height):
    pot = []
    for x in xs:
        if L/2 - l/2 < x < L/2 + l/2:
            pot.append(barrier_height)
        else:
            pot.append(0)
    return np.array(pot)


def simulate(sigX, barrier_height):
    gaussian = tools.create_gaussian(space, 5, sigX, k0)
    qmp = qmparticle.QMParticle(space, lambda x: gaussian, lambda x: potential(x, barrier_height))

    iterations = int(T / qmp.dt)

    animation_time = 10
    fps = 60
    frames = animation_time * fps

    Psi_real = [gaussian.real]
    Psi_imag = [gaussian.imag]
    rho_sq = [gaussian.real ** 2 + gaussian.imag ** 2]

    bar = Bar("Calculating Psi for sigX={0}, barrier={1}".format(sigX, barrier_height), max=frames)
    for i in range(frames):
        for j in range(iterations // frames):
            qmp.step()
        Psi_real.append(qmp.prev_Psi.real)
        Psi_imag.append(qmp.prev_Psi.imag)
        rho_sq.append(qmp.prev_Psi.real ** 2 + qmp.prev_Psi.imag ** 2)
        bar.next()
    bar.finish()

    pkl.dump([Psi_real, Psi_imag, rho_sq], open("problem4_sigX={0}_b={1}.p".format(sigX, barrier_height), "wb"))


def display(sigX, barrier_height):
    values = pkl.load(open("problem4_sigX={0}_b={1}.p".format(sigX, barrier_height), 'rb'))

    fig1, ax1 = plt.subplots(num=1)
    ax1.set_title(r"$\Psi$ after propagation", size=18)
    ax1.set_xlabel(r"$x$", size=18)
    #ax1.set_ylabel(r"$|\Psi|^2$", size=18)
    ax1.plot(space, values[0][-1], label=r"$\Psi_R$")
    ax1.plot(space, values[1][-1], label=r"$\Psi_I$")
    pot = potential(space, barrier_height)
    ax1.plot(space, (ax1.get_ylim()[1] / (2 * barrier_height))*pot, label=r"$V(x)$")
    plt.tight_layout()
    plt.legend()
    ax1.grid(True)

    plt.show()


def calculate_probabilities(sigX, barrier_height):
    values = pkl.load(open("problem4_sigX={0}_b={1}.p".format(sigX, barrier_height), 'rb'))
    rho_sq = values[2][-1]

    x_R = space[space <= L/2]
    R = np.trapz(rho_sq[:len(x_R)], x_R)

    x_T = space[space >= L/2]
    T = np.trapz(rho_sq[-len(x_T):], x_T)

    return (R, T)

def plot_it_ev0(barrier_heights):
    E_over_V0 = 1 / np.linspace(0, 3/2, 50, endpoint=True)[1:]
    probs = [calculate_probabilities(sigX, b) for b in barrier_heights[1:]]
    Rs, Ts = zip(*probs)

    plt.xlabel(r"$E/V_0$", size=18)
    plt.title("Probability of reflection and transmission", size=18)
    plt.plot(E_over_V0, Rs, label=r"$R$")
    plt.plot(E_over_V0, Ts, label=r"$T$")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def plot_it_v0e(barrier_heights):
    V0_over_E = np.linspace(0, 3/2, 50, endpoint=True)
    probs = [calculate_probabilities(sigX, b) for b in barrier_heights]
    Rs, Ts = zip(*probs)

    plt.xlabel(r"$V_0/E$", size=18)
    plt.title("Probability of reflection and transmission", size=18)
    plt.plot(V0_over_E, Rs, label=r"$R$")
    plt.plot(V0_over_E, Ts, label=r"$T$")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()


barrier_heights = np.linspace(0, 3/2, 50, endpoint=True) * k0**2 / 2

#for barrier_height in barrier_heights:
#    simulate(sigX, barrier_height)

plot_it_v0e(barrier_heights)






#simulate(sigX, k0**2 / 4)
#display(sigX, k0**2 / 4)
#calculate_probabilities(sigX, k0**2 / 4)

t1 = time.time()
print("Total time: {0} s".format(t1 - t0))
