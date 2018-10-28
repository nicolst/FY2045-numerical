import numpy as np
import threading


class QMParticle(threading.Thread):
    hbar = 1

    def __init__(self, x, initial_shape, potential, m=1,
                 name=None, stopping_condition=lambda x: True):
        self.m = m
        self.L = x[-1] - x[0]
        self.x = x

        self.prev_Psi = initial_shape(x)

        self.N = len(x)
        self.dx = self.L / self.N

        self.potential = potential(x)

        self.stopping_condition = stopping_condition

        # Set dt to reasonable value relative to dx and Vmax
        self.dt = 0.01 * self.hbar / \
            (np.linalg.norm(self.potential, np.inf) + self.hbar**2 / (2 * self.m * self.dx**2))

        self.main_const = self.hbar / (2 * self.m * self.dx**2)

        self.main_diag = -1 * (self.potential / self.hbar + 2 * self.main_const) * self.dt

        self.off_diag = self.main_const * self.dt

        # Set name to something resonable if not given
        if name is None:
            name = "QMpL{0}N{1}m{2}dx{3}dt{4}".format(self.L, self.N, self.m, self.dx, self.dt)
            
        threading.Thread.__init__(self, name=name)

    def run(self):
        while not self.stopping_condition:
            self.step()

    def step(self):
        new_Psi_imag = self.prev_Psi.imag + np.multiply(self.prev_Psi.real, self.main_diag) \
                       + np.multiply(np.roll(self.prev_Psi.real, 1), self.off_diag) \
                       + np.multiply(np.roll(self.prev_Psi.real, -1), self.off_diag)
        new_Psi_imag[-1] = 0
        new_Psi_imag[0] = 0

        new_Psi_real = self.prev_Psi.real - np.multiply(new_Psi_imag, self.main_diag) \
            - np.multiply(np.roll(new_Psi_imag, 1), self.off_diag) \
            - np.multiply(np.roll(new_Psi_imag, -1), self.off_diag)
        new_Psi_real[-1] = 0
        new_Psi_real[0] = 0

        self.prev_Psi = new_Psi_real + 1j * new_Psi_imag

