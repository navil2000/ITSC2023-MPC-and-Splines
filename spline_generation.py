from rbs_spline_module.spline_module import Rbs2_ControlPose, n_spline, t_spline
import numpy as np
from math import pi, sqrt
import matplotlib.pyplot as plt

if __name__ == "__main__":

  path = [	Rbs2_ControlPose(10, 40, -pi/4), 
            Rbs2_ControlPose(20, 20, 0), 
            Rbs2_ControlPose(40., 50, pi/4),
            Rbs2_ControlPose(60, 30, pi+pi/4), 
            Rbs2_ControlPose(40, 10, pi+pi/4)]

  t = []
  t_coeffs = []

  # Generate spline, and t-spline
  nu = 1
  u_coeffs = n_spline(path, nu)
  t, t_coeffs = t_spline(path, 1.0, u_coeffs)
  t2, t_coeffs2 = t_spline(path, 2.0, u_coeffs)

  # Some lists for plots
  Xt = []
  Yt = []
  Tt = []
  Xt2 = []
  Yt2 = []
  Tt2 = []
  Xu = []
  Yu = []

  # Spline data extraction for plots
  for coef in u_coeffs:
    ax, bx, cx, dx, ay, by, cy, dy  = coef
    u = np.linspace(0, 1, 100)
    x_spline = ax + bx*u + cx*u**2 + dx*u**3
    y_spline = ay + by*u + cy*u**2 + dy*u**3

    Xu.extend(x_spline)
    Yu.extend(y_spline)

  # T-Spline data extraction for plots
  for i, coef in enumerate(t_coeffs2):
    ax, bx, cx, dx, ay, by, cy, dy = coef
    t_segment = np.arange(t2[i], t2[i+1], 0.05)    
    x_spline = ax + bx*t_segment + cx*t_segment**2 + dx* t_segment**3
    y_spline = ay + by*t_segment + cy*t_segment**2 + dy* t_segment**3

    Xt2.extend(x_spline)
    Yt2.extend(y_spline)
    Tt2.extend(t_segment)

  # T-Spline data extraction for plots
  for i, coef in enumerate(t_coeffs):
    ax, bx, cx, dx, ay, by, cy, dy = coef
    t_segment = np.arange(t[i], t[i+1], 0.05)    
    x_spline = ax + bx*t_segment + cx*t_segment**2 + dx* t_segment**3
    y_spline = ay + by*t_segment + cy*t_segment**2 + dy* t_segment**3

    Xt.extend(x_spline)
    Yt.extend(y_spline)
    Tt.extend(t_segment)

  # 3D Graph
  fig = plt.figure()
  ax  = fig.add_subplot(111, projection='3d')

  # Add wayoints
  ax.scatter([p.x for p in path], [p.y for p in path], color='purple', label='Waypoints')

  # Add 3D representation for temporal splines and its projection in the plain X-Y
  ax.plot(Xt, Yt, Tt, color='green', label='T-spline Path (v=1)')
  ax.plot(Xt2, Yt2, Tt2, color='grey', label='T-spline Path (v=2)')
  ax.plot(Xu, Yu, color='red', label='Spline')

  # Add initial and last point of the path
  ax.scatter(Xt[0], Yt[0], Tt[0], color='blue')
  ax.scatter(Xt[-1], Yt[-1], Tt[-1], color='red')
  ax.scatter(Xt2[0], Yt2[0], Tt2[0], color='blue')
  ax.scatter(Xt2[-1], Yt2[-1], Tt2[-1], color='red')

  # Axis label and legend
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Time')
  ax.legend()
  plt.grid()

  # Save graph and show it
  plt.savefig("./Media/t-spline_representation.png")
  plt.show()