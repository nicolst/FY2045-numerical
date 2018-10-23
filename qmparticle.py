import matplotlib.pyplot as plt
from matplotlib import animation
import cupy as np
import threading
import tools
import numpy


class QMParticle(threading.Thread):
    hbar = 1

    def __init__(self, initial_shape: np.ndarray, x, dx=None, m=1, N=None,
                 dt=None, L: int=20, potential: np.ndarray=None,
                 name=None):
        self.m = m
        self.L = L
        self.x = x
        self.Psi = [initial_shape]

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

        self.main_diag = 1 - 1j * (self.potential / self.hbar + 2 * self.main_const) * self.dt
        self.main_diag[0] = 1
        self.main_diag[-1] = 1

        self.off_diag = np.full(self.N - 1, 1j * self.main_const * self.dt)
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
        new_Psi = self.Psi[-1] @ self.matrix_transform
        self.Psi.append(new_Psi)

#test = QMParticle(np.asarray([0, 0, 0]), N=7, potential=np.linspace(2, 8, 7, endpoint=True))

N = 2000
space, step_size = np.linspace(0, 20, N, endpoint=True, retstep=True)
gaussian = tools.create_gaussian(space, 5, 1.5, 20, step_size)
test = QMParticle(gaussian, space, step_size, N = N)

#plt.plot(np.asnumpy(space), np.asnumpy(test.Psi[0].real))

times = 100000
for i in range(times):
    test.step()
    print(i)

fig, ax = plt.subplots()
ln, = plt.plot(np.asnumpy(space), np.asnumpy(test.Psi[0].real))

def init():
    return ln,

def update(frame):
    ln.set_ydata(np.asnumpy(test.Psi[frame*times//600].real))
    return ln,

ani = animation.FuncAnimation(fig, update, frames=600, repeat=False, blit=False, interval=1/60)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)

ani.save("test.mp4", writer=writer)
#plt.draw()
#plt.show()