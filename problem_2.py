import qmparticle
import numpy as np
import matplotlib.pyplot as plt
import tools
import time


t0 = time.time()

L = 20
N = 2000
k0 = 20

space, step_size = np.linspace(0, L, N, endpoint=True, retstep=True)
gaussian = tools.create_gaussian(space, 5, 1.5, k0, step_size)
test = qmparticle.QMParticle(space, lambda x: gaussian, lambda x: x * 0)
print("Starting loop")
for i in range(int(0.5/(1000 * test.dt))):
    for j in range(1000):
        test.step()
    print(i)

t1 = time.time()
print("Took {0} seconds!".format(t1 - t0))

plt.plot(space, test.prev_Psi)
plt.show()
