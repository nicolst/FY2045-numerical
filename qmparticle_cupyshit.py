import cupy as np
import threading


class QMParticle(threading.Thread):
    hbar = 1

    def __init__(self, x, initial_shape, potential, m=1,
                 name=None):
        self.m = m
        self.L = x[-1] - x[0]
        self.x = x

        self.prev_Psi = initial_shape(x)

        self.N = len(x)
        self.dx = self.L / self.N

        self.potential = potential(x)

        # Set dt to reasonable value relative to dx and Vmax
        self.dt = 0.01 * self.hbar / \
            (np.linalg.norm(self.potential, np.inf) + self.hbar**2 / (2 * self.m * self.dx**2))

        self.main_const = self.hbar / (2 * self.m * self.dx**2)

        self.main_diag = -1 * (self.potential / self.hbar + 2 * self.main_const) * self.dt
        self.main_diag[0] = 1
        self.main_diag[-1] = 1

        self.off_diag = np.full(self.N - 1, self.main_const * self.dt)
        self.off_diag[0] = 0
        self.off_diag[-1] = 0

        self.matrix_transform = np.diag(self.off_diag, -1) + np.diag(self.main_diag, 0) + np.diag(self.off_diag, 1)


        # Set name to something resonable if not given
        if name is None:
            name = "QM particle (m={0}, N={1}, dt={2}, L={3}".format(self.m, self.N, self.dt, self.L)
            
        threading.Thread.__init__(self, name=name)

    def run(self):
        pass

    def step(self):
        new_Psi_imag = self.prev_Psi.imag + (self.prev_Psi.real @ self.matrix_transform)
        new_Psi_real = self.prev_Psi.real - (new_Psi_imag @ self.matrix_transform)
        self.prev_Psi = new_Psi_real + 1j * new_Psi_imag

