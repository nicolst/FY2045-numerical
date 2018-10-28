import cupy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pickle as pkl
from qmparticle_cupyshit import QMParticle
import tools

#test = QMParticle(np.asarray([0, 0, 0]), N=7, potential=np.linspace(2, 8, 7, endpoint=True))

N = 2000
L = 20
k0 = 20
m = 1
space, step_size = np.linspace(0, L, N, endpoint=True, retstep=True)
gaussian = tools.create_gaussian(space, 5, 1.5, k0, step_size)
pot0 = [0.0] * int(0.48*N)
pot = np.asarray(pot0 + [QMParticle.hbar**2 * k0**2 / (4 * m)]*(N - 2 * len(pot0)) + pot0)
#test = QMParticle(gaussian, space, step_size, N = N, potential=pot)
test = QMParticle(gaussian, space, step_size, N = N)

#plt.plot(np.asnumpy(space), np.asnumpy(test.Psi[0].real))

times = int(0.5/test.dt)
print("Iterating {0} times with {1} steps per".format(600, times//600))
for i in range(600):
    test.step_memsave(times//600)
    print(i)

def save():
    pkl.dump(test.Psi, open('Psi.p', 'wb'))
    pkl.dump(test.rho_sq, open('rho_sq.p', 'wb'))
save()

fig, ax = plt.subplots()
ln1, = plt.plot(np.asnumpy(space), np.asnumpy(test.Psi[0].real), label="Re")
ln2, = plt.plot(np.asnumpy(space), np.asnumpy(test.Psi[0].imag), label="Im")
ln3, = plt.plot(np.asnumpy(space), np.asnumpy(test.rho_sq[0]), label=r"$\rho^2$")
plt.plot(np.asnumpy(space), np.asnumpy(pot * np.linalg.norm(test.Psi[0].real, np.inf) / (2 * np.linalg.norm(pot, np.inf))))
plt.legend()


def update(frame):
    i = frame * times // 600
    ln1.set_ydata(np.asnumpy(test.Psi[i].real))
    ln2.set_ydata(np.asnumpy(test.Psi[i].imag))
    ln3.set_ydata(np.asnumpy(test.rho_sq[i]))
    return ln1,ln2,ln3

def update_memsave(frame):
    ln1.set_ydata(np.asnumpy(test.Psi[frame].real))
    ln2.set_ydata(np.asnumpy(test.Psi[frame].imag))
    ln3.set_ydata(np.asnumpy(test.rho_sq[frame]))
    return ln1,ln2,ln3

ani = animation.FuncAnimation(fig, update, frames=600, repeat=False, blit=False, interval=1/60)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)

ani.save("test.mp4", writer=writer)
#plt.draw()
#plt.show()