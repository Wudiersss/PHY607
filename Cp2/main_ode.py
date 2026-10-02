import numpy as np
import matplotlib.pyplot as plt
from ode import el, rk4, sp, exact


# Parameters and initial
c1 = 2.0
c2 = 3.0
k = 0.5
y0 = np.array([100.0, 20.0])
t = np.linspace(0, 20, 101)

# Run
ye = el(t, y0, c1, c2, k)
yr = rk4(t, y0, c1, c2, k)
ys = sp(t, y0, c1, c2, k)
yx = exact(t, y0, c1, c2, k)

print("ODE example: temperatures at t = 20")
print("Euler: ", ye[-1])
print("RK4:   ", yr[-1])
print("SciPy: ", ys[-1])
print("Exact: ", yx[-1])

# Compare with exact answer
ee = np.max(np.abs(ye - yx))
er = np.max(np.abs(yr - yx))
es = np.max(np.abs(ys - yx))
print("\nMaximum errors")
print("Euler:", ee)
print("RK4:  ", er)
print("SciPy:", es)

# Self check 1 energy
energy_rk = c1 * yr[:, 0] + c2 * yr[:, 1]
print("\nSelf check 1: energy conservation")
print("RK4 energy change:  ", np.max(np.abs(energy_rk - energy_rk[0])))

# Self check 2 equilibrium
print("\nSelf check 2: thermal equilibrium")
print("Initial temperature difference:", abs(y0[0] - y0[1]))
print("Final temperature difference:  ", abs(yr[-1, 0] - yr[-1, 1]))

# Plot
plt.figure()
plt.plot(t, ye[:, 0], "--", label="Euler T1")
plt.plot(t, yr[:, 0], label="RK4 T1")
plt.plot(t, ys[:, 0], ":", label="SciPy T1")
plt.plot(t, yx[:, 0], "k", label="Exact T1")
plt.plot(t, yr[:, 1], label="RK4 T2")
plt.plot(t, yx[:, 1], "k--", label="Exact T2")
plt.xlabel("time")
plt.ylabel("temperature")
plt.title("Two objects exchanging heat")
plt.legend()
plt.savefig("ode.png")
plt.close()

# Error scaling
ns = [21, 41, 81, 161, 321]
hs = []
err_el = []
err_rk = []

for n in ns:
    tn = np.linspace(0, 20, n)
    yn = exact(tn, y0, c1, c2, k)
    hs.append(tn[1] - tn[0])
    err_el.append(np.max(np.abs(el(tn, y0, c1, c2, k) - yn)))
    err_rk.append(np.max(np.abs(rk4(tn, y0, c1, c2, k) - yn)))

print("\nODE error scaling")
for i in range(len(ns)):
    print("h =", hs[i], " Euler error =", err_el[i],
          " RK4 error =", err_rk[i])

plt.figure()
plt.loglog(hs, err_el, "o-", label="Euler")
plt.loglog(hs, err_rk, "s-", label="RK4")
plt.xlabel("step size h")
plt.ylabel("maximum error")
plt.title("ODE error scaling")
plt.legend()
plt.savefig("ode_error.png")
plt.close()
