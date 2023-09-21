import matplotlib.pyplot as plt
import numpy as np

from rbs_mpc import rbs_mpc 
from scipy import sparse

if __name__ == "__main__":
    
    # Simulation parameters
    nsim = 200
    Ts = 0.1
    # MPC Hyper-parameters
    q11 = 100000
    q22 = 100
    r   = 0.1
    
    Q = sparse.diags([q11, q22])  
    R = r*sparse.eye(1) 
    
    # Model matrices, constraints and more
    Ad = sparse.csc_matrix([[1, Ts],
                        [0, 1]])

    Bd = sparse.csc_matrix([[(Ts**2/2)],
                            [ Ts]])
   
    u0 = 0
    umin = np.array([-10])
    umax = np.array([10])
    x0 = np.array([0, 0])
    xmin = np.array([-8, -8])
    xmax = np.array([8, 8])
   
    # State vector reference
    xr = np.array([0.0, 0.0])           
    
    # Prediction horizont
    N = 50 

    # MPC object
    mpc_lateral = rbs_mpc(A=Ad,B=Bd,
                            u0=u0,umin=umin,umax=umax,
                            x0=x0,xmin=xmin,xmax=xmax,
                            Q=Q,R=R,N=N,xr=xr)
    
    # Lateral deviation reference
    de_ref = 0

    # Some list for plots
    x_list = []
    r_list = []

    for i in range(nsim):

        # Step 70 -> Change reference
        if i > 70 and i < 140:
            de_ref = 3.5
        else:
            de_ref = 0

        # Update lateral mpc reference 
        mpc_lateral.set_x_ref(np.array([de_ref, 0.0]))    
        # Solve the mpc
        ctrl = mpc_lateral.mpc_move(x0)
        # Plant simulation
        x0 = Ad.dot(x0) + Bd.dot(ctrl)

        r_list.append(de_ref)
        x_list.append(x0)

    tiempo = [Ts*i for i in range(nsim)]

    plt.plot(tiempo, x_list)
    plt.plot(tiempo, r_list)
    plt.ylabel(f'de')
    plt.xlabel(f'time')
    plt.legend([f'de', f'v_lat', f'Reference'])
    plt.title(f'q11 = {q11}, q22 = {q22}, r = {r}')
    plt.savefig("../Media/mpc_test_lateral_model.png")
    plt.show()
