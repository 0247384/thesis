from scipy.odr import odr
from scipy.stats import linregress
from matplotlib.pyplot import *


# matplotlib.rcParams.update({'font.size': 18})


# Source: docs.scipy.org/doc/scipy/reference/odr.html
def linear(b, x):
    return b[0] * x + b[1]


mean = (2.5, 2.5)
cov = [[1.5, 0.3], [0.7, 1.5]]
data = np.random.multivariate_normal(mean, cov, 10)

x = [max(0.5, min(s[0], 4.5)) for s in data]
y = [max(0.5, min(s[1], 4.5)) for s in data]

odr_result = odr(linear, [0, 0], y, x, full_output=1)
m, b = odr_result[0]
m = round(m, 6)
b = round(b, 6)
f_odr = np.poly1d((m, b))

ols_result = linregress(x, y)
m, b = ols_result[0:2]
m = round(m, 6)
b = round(b, 6)
f_ols = np.poly1d((m, b))

r = [-1, 6]

fig = figure(figsize=(14, 7))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(x, y, 'bo', markersize=10)
ax1.plot(r, f_ols(r), 'g', linewidth=2)
ax1.set_aspect('equal')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
# ax1.set_title('OLS')
# ax1.grid('on')
ax1.set_xlim([0, 5])
ax1.set_ylim([0, 5])

for i in range(len(x)):
    ax1.plot([x[i], x[i]], [y[i], f_ols(x[i])], 'r--', linewidth=1)

ax2.plot(x, y, 'bo', markersize=10)
ax2.plot(r, f_odr(r), 'g', linewidth=2)
ax2.set_aspect('equal')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
# ax2.set_title('ODR')
# ax2.grid('on')
ax2.set_xlim([0, 5])
ax2.set_ylim([0, 5])

for i in range(len(x)):
    m, b = odr_result[0]
    v_m = np.array([1, m])
    v_b = np.array([0, b])
    v_p = np.array([x[i], y[i]])
    v_pb = v_m * (np.dot(v_b, v_m) / np.dot(v_m, v_m))
    v_d = v_b - v_pb
    v_pp = v_m * (np.dot(v_p, v_m) / np.dot(v_m, v_m))
    v_pp = v_pp + v_d
    ax2.plot([x[i], v_pp[0]], [y[i], v_pp[1]], 'r--', linewidth=1)

show()
