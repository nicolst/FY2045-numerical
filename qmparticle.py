import matplotlib.pyplot as plt
from matplotlib import animation
import cupy as np
import threading
import tools
import numpy
import pickle as pkl


class QMParticle(threading.Thread):
    hbar = 1

    def __init__(self, initial_shape: np.ndarray, x, dx=None, m=1, N=None,
                 dt=None, L: int=20, potential: np.ndarray=None,
                 name=None):
        self.m = m
        self.L = L
        self.x = x
        self.Psi = [np.asnumpy(initial_shape)]
        self.rho_sq = [np.asnumpy(initial_shape.real**2 + initial_shape.imag**2)]

        self.prev_Psi = initial_shape

        if dx is None:
            self.dx = x[1] - x[0]
        else:
            self.dx = dx

        print(x[1], type(x[1]))
        # If N not given, set to ~100 times the length of the space
        if N is None:
            self.N = round(float(x[-1] - x[0]) / dx) + 1
        else:
            self.N = N

        print(self.N, len(self.x))

        # If potential not given, set it to a zero potential
        if potential is None:
            self.potential = np.zeros(self.x.shape)
        else:
            self.potential = potential

        # If dt not given, set to reasonable value relative to dx and Vmax
        if dt is None:
            self.dt = 0.01 * self.hbar / \
                      (np.linalg.norm(self.potential, np.inf) + self.hbar ** 2 / (2 * self.m * self.dx ** 2))
        else:
            self.dt = dt

        self.main_const = self.hbar / (2 * self.m * self.dx**2)

        self.main_diag = -1 * (self.potential / self.hbar + 2 * self.main_const) * self.dt
        self.main_diag[0] = 1
        self.main_diag[-1] = 1

        self.off_diag = np.full(self.N - 1, self.main_const * self.dt)
        self.off_diag[0] = 0
        self.off_diag[-1] = 0

        print(len(np.diag(self.off_diag, -1)), len(np.diag(self.main_diag, 0)))

        self.matrix_transform = np.diag(self.off_diag, -1) + np.diag(self.main_diag, 0) + np.diag(self.off_diag, 1)


        # Set name to something resonable if not given
        if name is None:
            name = "QM particle (m={0}, N={1}, dt={2}, L={3}"
            
        threading.Thread.__init__(self, name=name)

    def run(self):
        pass

    def step(self):
        new_Psi_imag = self.prev_Psi.imag + (self.prev_Psi.real @ self.matrix_transform)
        new_Psi_real = self.prev_Psi.real - (new_Psi_imag @ self.matrix_transform)
        self.Psi.append(np.asnumpy(new_Psi_real + 1j * new_Psi_imag))
        self.prev_Psi = new_Psi_real + 1j * new_Psi_imag
        self.rho_sq.append(np.asnumpy(new_Psi_real**2 + new_Psi_imag**2))

#test = QMParticle(np.asarray([0, 0, 0]), N=7, potential=np.linspace(2, 8, 7, endpoint=True))

N = 2000
L = 20
k0 = 20
m = 1
space, step_size = np.linspace(0, L, N, endpoint=True, retstep=True)
gaussian = tools.create_gaussian(space, 5, 1.5, k0, step_size)
pot0 = [0.0] * int(0.48*N)
pot = np.asarray(pot0 + [QMParticle.hbar**2 * k0**2 / (4 * m)]*(N - 2 * len(pot0)) + pot0)
test = QMParticle(gaussian, space, step_size, N = N, potential=pot)

#plt.plot(np.asnumpy(space), np.asnumpy(test.Psi[0].real))

times = 300000
for i in range(times):
    test.step()
    print(i)

#pkl.dump(test, open('test.p', 'wb'))

fig, ax = plt.subplots()
ln1, = plt.plot(np.asnumpy(space), test.Psi[0].real, label="Re")
ln2, = plt.plot(np.asnumpy(space), test.Psi[0].imag, label="Im")
ln3, = plt.plot(np.asnumpy(space), test.rho_sq[0], label=r"$\rho^2$")
plt.plot(np.asnumpy(space), np.asnumpy(pot * np.linalg.norm(test.Psi[0].real, np.inf) / (2 * np.linalg.norm(pot, np.inf))))
plt.legend()


def update(frame):
    i = frame * times // 600
    ln1.set_ydata(np.asnumpy(test.Psi[i].real))
    ln2.set_ydata(np.asnumpy(test.Psi[i].imag))
    ln3.set_ydata(np.asnumpy(test.rho_sq[i]))
    return ln1,ln2,ln3

ani = animation.FuncAnimation(fig, update, frames=600, repeat=False, blit=False, interval=1/60)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)

ani.save("test.mp4", writer=writer)
#plt.draw()
#plt.show()