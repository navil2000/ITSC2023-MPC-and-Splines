# Importamos las librerías necesarias.
import osqp
import numpy as np
import scipy as sp
from scipy import sparse

class rbs_mpc:
    def __init__(self, A, B, u0, umin, umax, x0, xmin, xmax, Q, R, N, xr):
        
        # Recogemos las variables
        self.A = A
        self.B = B
        self.u0 = u0
        self.umin = umin 
        self.umax = umax
        self.x0 = x0
        self.xmin = xmin
        self.xmax = xmax
        self.Q = Q
        self.QN = self.Q
        self.R = R
        self.N = N
        self.xr = xr

        #Comenzamos a configurar el MPC

        # Definimos las dimensiones de nuestro problema en función a B
        [self.nx, self.nu] = self.B.shape

        # Expadimos el sistema de entrada para ajustarlo a un MPC
        # tal que X = (x(0),...,x(N),u(0),...,u(N-1))

        # Objetivo cuadrático
        self.P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), self.QN,
                       sparse.kron(sparse.eye(N), R)], format='csc') 
        # Objetivo lineal
        self.q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -self.QN.dot(xr),
               np.zeros(N*self.nu)])

        # Dinámica Lineal
        self.Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(self.nx)) + sparse.kron(sparse.eye(N+1, k=-1), A)
        self.Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), B)
        self.Aeq = sparse.hstack([self.Ax, self.Bu])
        self.leq = np.hstack([-x0, np.zeros(N*self.nx)])
        self.ueq = self.leq
        # Redefinimos las restricciones
        self.Aineq = sparse.eye((N+1)*self.nx + N*self.nu)
        self.lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
        self.uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
        self.Af = sparse.vstack([self.Aeq, self.Aineq], format='csc')
        self.l = np.hstack([self.leq, self.lineq])
        self.u = np.hstack([self.ueq, self.uineq])
        
        # Creamos el objeto de la clase problema OSQP y lo inicializamos
        self.problema = osqp.OSQP()
        self.problema.setup(P=self.P,q=self.q,A=self.Af,l=self.l,u=self.u, warm_start=True, verbose=False)
        
    def set_u_limits(self, umin, umax):
        self.umax = umax
        self.umin = umin
        self.lineq = np.hstack([np.kron(np.ones(self.N+1), self.xmin), np.kron(np.ones(self.N), self.umin)])
        self.uineq = np.hstack([np.kron(np.ones(self.N+1), self.xmax), np.kron(np.ones(self.N), self.umax)])
        self.l = np.hstack([self.leq, self.lineq])
        self.u = np.hstack([self.ueq, self.uineq])
        self.problema.update(l=self.l, u=self.u)         

    def set_x_limits(self, xmin, xmax):
        self.xmax = xmax
        self.xmin = xmin
        self.lineq = np.hstack([np.kron(np.ones(self.N+1), self.xmin), np.kron(np.ones(self.N), self.umin)])
        self.uineq = np.hstack([np.kron(np.ones(self.N+1), self.xmax), np.kron(np.ones(self.N), self.umax)])
        self.l = np.hstack([self.leq, self.lineq])
        self.u = np.hstack([self.ueq, self.uineq])
        self.problema.update(l=self.l, u=self.u)        

    def set_x_ref(self, xref):
        self.xr = xref 
        self.q = np.hstack([np.kron(np.ones(self.N), -self.Q.dot(self.xr)), -self.QN.dot(self.xr),
               np.zeros(self.N*self.nu)])
        self.problema.update(q=self.q)

    def mpc_move(self, x0):
        
        # Feedback de la planta para actualizar el problema.
        self.x0 = x0
        self.l[:self.nx] = -self.x0
        self.u[:self.nx] = -self.x0
        self.problema.update(l=self.l, u=self.u)

        # Resolvemos el sistema
        self.res = self.problema.solve()

        # Check de que hayamos podido solucionar el problema 
        if self.res.info.status != 'solved':
            print('No solution found for the given conditions')

        return self.res.x[-self.N*self.nu:-(self.N-1)*self.nu]