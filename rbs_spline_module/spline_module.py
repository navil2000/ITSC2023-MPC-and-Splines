from math import cos, sin, pi, sqrt
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Define a class for Rbs2_ControlPose for clarity
class Rbs2_ControlPose:
    def __init__(self, x, y, o):
        self.x = x
        self.y = y
        self.o = o

# Spline
def n_spline(waypoints_path, nu, lateral_offsets=[]):
  '''
  Input:
      · waypoints: (X,Y,O).
      · nu: smoothing factor
      · lateral_offsets: list of lateral offsets for each waypoint. 
      If it is None, no lateral offsets are applied (offset = 0).

  Output:
      · coeffs: List of tuples defining the polynomials.

  '''
  waypoints = waypoints_path[:]
  offsets = lateral_offsets[:]

  m = len(waypoints)
  lam = []
  deta = []
  D = []
  coefs = []

  if len(offsets) != 0:
    for offset, waypoint in zip(lateral_offsets, waypoints):
      waypoint.x = waypoint.x + offset*sin(waypoint.o)
      waypoint.y = waypoint.y - offset*cos(waypoint.o)

  # X(u) #
  lam.append(0)
  deta.append(nu*cos(waypoints[0].o))

  for i in range(1, m - 1):
    lam.append(1 / (4 - lam[i - 1]))
    deta.append((3 * (waypoints[i + 1].x - waypoints[i - 1].x) - deta[i - 1]) * lam[i])
  deta.append(nu*cos(waypoints[-1].o))

  for _ in range(m):
    D.append(0)

  D[m - 1] = deta[m - 1]

  for i in range(m - 2, 0, -1):
    D[i] = deta[i] - lam[i] * D[i + 1]

  for i in range(m - 1):
    coef = [
      waypoints[i].x,  # Add lateral offset
      D[i],
      3 * (waypoints[i + 1].x - waypoints[i].x) - 2 * D[i] - D[i + 1],
      2 * (waypoints[i].x - waypoints[i + 1].x) + D[i] + D[i + 1],
    ]
    coefs.append(coef)

  # Y(u) #
  lam[0] = 0
  deta[0] = nu*sin(waypoints[0].o)

  for i in range(1, m - 1):
    lam[i] = 1 / (4 - lam[i - 1])
    deta[i] = ((3 * (waypoints[i + 1].y - waypoints[i - 1].y) - deta[i - 1]) * lam[i])
  deta[-1] = nu * sin(waypoints[-1].o)

  D[m - 1] = deta[m - 1]

  for i in range(m - 2, -1, -1):
    D[i] = deta[i] - lam[i] * D[i + 1]

  for i in range(m - 1):
    coef = [
        waypoints[i].y,  # Add lateral offset
        D[i],
        3 * (waypoints[i + 1].y - waypoints[i].y) - 2 * D[i] - D[i + 1],
        2 * (waypoints[i].y - waypoints[i + 1].y) + D[i] + D[i + 1],
    ]
    coefs[i].extend(coef)

  return coefs

# T-spline
def t_spline(waypoints_path, vel, coeffs_spline):
  '''
  Generates time-parameterized splines.
  Input:
      · waypoints: (X,Y,O).
      · nu: smoothing factor
      · vel: object/vehicle velocity
      
  Output:
      · t: list with the time boundaries of each segment
      ·t_coeffs: coefficients of the parameterized polynomial
  '''

  # Aux variables
  waypoints = waypoints_path[:]
  coeffs = coeffs_spline[:]
  m = len(waypoints)
  t = []
  inc = []
  t_coeffs = []

  # Euclidean distances between waypoints
  dist = [sqrt((waypoints[i+1].x-waypoints[i].x)**2 
              + (waypoints[i+1].y-waypoints[i].y)**2) for i in range(m-1)]

  # Vector of time boundaries
  k = 1 / vel
  t.append(0)
  for i in range(1, m):
    t.append(k * dist[i-1] + t[i-1])

  # Time base and variation
  b=t
  for i in range(m-1):
    inc.append(t[i+1]-t[i])

  # T-Coeffs
  for i in range(m-1):  
    # For each segment
    coef = [coeffs[i][0] - (b[i]**3 * coeffs[i][3] / inc[i]**3) + (b[i]**2 * coeffs[i][2] / inc[i]**2) - (b[i] * coeffs[i][1] / inc[i]),
            (coeffs[i][1] / inc[i]) + (3 * b[i]**2 * coeffs[i][3] / inc[i]**3) - (2 * b[i] * coeffs[i][2] / inc[i]**2),
            (-3 * b[i] * coeffs[i][3] / inc[i]**3) + (coeffs[i][2] / inc[i]**2),
            coeffs[i][3] / inc[i]**3,
            coeffs[i][4] - (b[i]**3 * coeffs[i][7] / inc[i]**3) + (b[i]**2 * coeffs[i][6] / inc[i]**2) - (b[i] * coeffs[i][5] / inc[i]),
            (coeffs[i][5] / inc[i]) + (3 * b[i]**2 * coeffs[i][7] / inc[i]**3) - (2 * b[i] * coeffs[i][6] / inc[i]**2),
            (-3 * b[i] * coeffs[i][7] / inc[i]**3) + (coeffs[i][6] / inc[i]**2),
            coeffs[i][7] / inc[i]**3]

    t_coeffs.append(coef)

  return t, t_coeffs

