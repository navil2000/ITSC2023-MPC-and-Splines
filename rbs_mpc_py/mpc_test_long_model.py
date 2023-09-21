from rbs_mpc import rbs_mpc
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

if __name__ == "__main__":

	# Simulation parameters
	Ts = 0.1
	nsim = 500

	# Model matrices, constraints and more
	Ad = sparse.csc_matrix([[1, Ts, (Ts**2/2)],
							[0, 1,  Ts],
							[0, 0,  1]])

	Bd = sparse.csc_matrix([[(Ts**3/6)],
							[(Ts**2/2)],
							[Ts]])

	u0 = 0
	umin = np.array([-10])
	umax = np.array([10])

	x0 = np.array([5, 0, 0])
	xmin = np.array([-8, -2, -8])
	xmax = np.array([8,  2,  8])

	xr = np.array([-2, 0.0, 0.0])

	# MPC Hyper-parameters
	Q = sparse.diags([1000, 5000, 1000])
	R = 500*sparse.eye(1)

	# Prediction horizont
	N = 50 

	# MPC object
	mpc_lateral = rbs_mpc(A=Ad,B=Bd,
						u0=u0,umin=umin,umax=umax,
						x0=x0,xmin=xmin,xmax=xmax,
						Q=Q,R=R,N=N,xr=xr)

	# Some list for plots
	x_list = []
	r_list = []

	for _ in range(nsim):
		# Solve the mpc
		ctrl = mpc_lateral.mpc_move(x0)
		# Plant simulation
		x0 = Ad.dot(x0) + Bd.dot(ctrl)

		r_list.append(xr[0])
		x_list.append(x0)

	tiempo = [Ts*i for i in range(nsim)]

	plt.figure(figsize=(10, 6))
	plt.plot(tiempo, x_list)
	plt.plot(tiempo, r_list, color="purple")
	plt.ylabel(f'de')
	plt.xlabel(f'tiempo')
	plt.legend([f'd_lon', f'v_lon', f'a_lon', f'Reference'])
	plt.title(f'Q = {Q.todense()} \n- R = {R.todense()}')
	plt.savefig("../Media/mpc_test_long_model.png")
	plt.show()
