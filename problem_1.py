import qmparticle
import numpy as np
import tools
import matplotlib.pyplot as plt

L = 20
N = 2000
k0 = 20
T = 0.5
sigX = 1.5

space = np.linspace(0, L, N, endpoint=True)

gaussian = tools.create_gaussian(space, 5, sigX, k0)

plt.figure(1)
plt.xlabel(r"$x$", size=18)
plt.plot(space, gaussian.real, label=r"$\Psi_R$")
plt.plot(space, gaussian.imag, label=r"$\Psi_I$")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(2)
plt.xlabel(r"$x$", size=18)
plt.ylabel(r"$|\Psi|^2$", size=18)
plt.plot(space, gaussian.real**2 + gaussian.imag**2)
plt.grid(True)
plt.tight_layout()

plt.show()
