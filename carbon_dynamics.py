import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import pdb
import colorcet as cc
from typing import Callable

''' Equilibrium mass C relative to atmosphere ''' 
N_a = 1
N_b = 0.5
N_d = 1.5
N_m = 1.2
N_s = 60

def solve(T: float, dt: float=1., gamma: Callable=lambda t: 7.5, M_a=820, tau_am=5, tau_sm=500, tau_ab=30, tau_bd=30):
	'''
	T: extent (years)
	dt: time step
	gamma: emission function, in Pg C / Y
	M_a: initial mass C in atmosphere, in Pg
	tau_ij: mean first passage time from reservoir i to j, in Y
	'''

	''' Equilibrium const. in Y^-1 ''' 

	k_am = 1/tau_am
	k_sm = 1/tau_sm
	k_ab = 1/tau_ab
	k_bd = 1/tau_bd

	A = np.array([
		[N_m, 0, N_b, N_d],
		[0, 0, -N_b, 0],
		[0, 0, 0, -N_d],
		[-N_m, -N_m, 0, 0],
		[0, N_m, 0, 0]
	])
	b = np.array([
		k_am*N_a + k_ab*N_a,
		-k_ab*N_a + k_bd*N_b,
		-k_bd*N_b,
		-k_am*N_a - k_sm*N_s,
		k_sm*N_s
	])

	[k_ma, k_ms, k_ba, k_da] = nnls(A, b)[0]

	''' ODE '''

	# Vector order: a, b, d, m, s
	def f(t, n):
		dndt = np.zeros_like(n)
		dndt[0] = -k_am*n[0] + k_ma*n[3] - k_ab*n[0] + k_ba*n[1] + k_da*n[2] + gamma(t)
		dndt[1] = k_ab*n[0] - k_ba*n[1] - k_bd*n[1]
		dndt[2] = -k_da*n[2] + k_bd*n[1]
		dndt[3] = k_am*n[0] - k_ma*n[3] - k_ms*n[3] + k_sm*n[4]
		dndt[4] = k_ms*n[3] - k_sm*n[4]
		return dndt

	n0 = np.array([M_a, M_a*N_b, M_a*N_d, M_a*N_m, M_a*N_s])
	t_eval = np.linspace(0., T, int(T/dt))
	sol = solve_ivp(f, (0., T), n0, t_eval=t_eval)

	return sol.t, n0, sol.y

plots = {
	0: 'Atmosphere',
	1: 'Biosphere',
	2: 'Detritus', 
	3: 'Mixed Layer',
	4: 'Deep Sea',
}

tau_am_set = [2, 5, 10]
T = 1000.

fig, axs = plt.subplots(nrows=5, ncols=1, figsize=(6,10))
for j, tau_am in enumerate(tau_am_set):
	t, n0, n = solve(T, tau_am=tau_am)
	for (i, title) in plots.items():
		pct_change = 100 * (n[i] - n0[i]) / n0[i]
		axs[i].plot(t, pct_change, label=f'tau_am={tau_am}', color=cc.glasbey[j])
		axs[i].set_title(title, fontsize=10)
		axs[i].legend()
fig.suptitle(f'% Change C at {int(T)} years', fontsize=14)
plt.tight_layout()

plt.show()