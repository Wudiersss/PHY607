import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from integ import gauss, gauss_exact, left, trap, simp


# Parameters
a = 2.0
mu = 1.0
length = 1.5
lo = mu - length
hi = mu + length
n = 20          # Shoule be even! For Simpson
ans = gauss_exact(lo, hi, a, mu)

# Run hand-written and SciPy methods
v_left = left(gauss, lo, hi, n, a, mu)
v_trap = trap(gauss, lo, hi, n, a, mu)
v_simp = simp(gauss, lo, hi, n, a, mu)
v_sp = quad(gauss, lo, hi, args=(a, mu))[0]

print("Integral example: parameterized Gaussian")
print("Left:         ", v_left)
print("Trapezoid:    ", v_trap)
print("Simpson:      ", v_simp)
print("SciPy quad:   ", v_sp)
print("Exact:        ", ans)

# Self check 1: moving the center and limits together changes nothing
shift = 4.0
original = simp(gauss, lo, hi, 200, a, mu)
moved = simp(gauss, lo + shift, hi + shift, 200, a, mu + shift)
print("\nSelf check 1: translation difference:", abs(original - moved))

# Self check 2: change variable by scaling u = sqrt(a) * (x - mu)
new_lo = -np.sqrt(a) * length
new_hi = np.sqrt(a) * length
scaled = simp(gauss, new_lo, new_hi, 200, 1.0, 0.0) / np.sqrt(a)
print("Self check 2: scaling difference:    ", abs(original - scaled))


# Error scaling
ns = [10, 20, 40, 80, 160, 320]
hs = []
err_left = []
err_trap = []
err_simp = []

lo_err = -1.0
hi_err = 2.0
a_err = 1.0
mu_err = 0.0
ans_err = gauss_exact(lo_err, hi_err, a_err, mu_err)

for n in ns:
    hs.append((hi_err - lo_err) / n)
    err_left.append(abs(left(gauss, lo_err, hi_err, n, a_err, mu_err) - ans_err))
    err_trap.append(abs(trap(gauss, lo_err, hi_err, n, a_err, mu_err) - ans_err))
    err_simp.append(abs(simp(gauss, lo_err, hi_err, n, a_err, mu_err) - ans_err))

print("\nIntegral error scaling")
for i in range(len(ns)):
    print("n =", ns[i], " Left error =", err_left[i],
          " Trap error =", err_trap[i], " Simpson error =", err_simp[i])

plt.figure()
plt.loglog(hs, err_left, "o-", label="Left")
plt.loglog(hs, err_trap, "s-", label="Trapezoid")
plt.loglog(hs, err_simp, "^-", label="Simpson")
plt.xlabel("step size h")
plt.ylabel("absolute error")
plt.title("Integral error scaling")
plt.legend()
plt.savefig("integral_error.png")
plt.close()

