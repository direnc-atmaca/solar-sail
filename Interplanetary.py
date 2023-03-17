
'''
Author: Direnc Atmaca
Date of comment: March 17, 2023

Purpose: This code simulates the Interplanetary section of the Mission
'''

# Importing necessary packages for numerical simulations, math, and plotting
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp


tu = 806.8; du = 6378.136  # canonical units set in a way to lead mu = 1
AU = 149597870.691  # 1 AU or the distance between the Earth and Sun in km
eps = math.radians(23.45)  # Ecliptic tilt

Re = 6378.136/du  # Radius of the Earth
mu = 1  # Gravitational parameter of the Earth in canonical unit

mu_sun = (1.327124*(10**11))*((tu**2)/(du**3))  # Gravitational parameters of the Sun in canonical units
we = (2*np.pi)/(86164/tu)  # Rotation rate of the Earth around its axis

# Initial orbit definition in polar coordinates
r0 = AU/du  # initial orbital radius from the Sun
theta0 = 0  # initial polar angle

'''
Parameters below are the lightness number (beta), sail pitch angle (alpha), and spiral angle (gamma)
To run the simulation appropriate values must be uncommented while all others remain commented
'''

# Parameters of the Sail
beta = 0.015  # lightness number
alpha = np.radians(35.19)  # sail pitch angle
gamma = np.radians(1.26)  # spiral angle

#beta = 0.02
#alpha = np.radians(35.16)
#gamma = np.radians(1.46)

#beta = 0.05
#alpha = np.radians(34.98)
#gamma = np.radians(2.66)

#beta = 0.1
#alpha = np.radians(34.68)
#gamma = np.radians(4.66)

#beta = 0.125
#alpha = np.radians(34.53)
#gamma = np.radians(5.90)

#beta = 0.15
#alpha = np.radians(34.37)
#gamma = np.radians(7.18)


# Differential System
def dSdt(t, S):

    r, theta = S
    # 2D Logarithmic dynamics - depends on the solar sail performance parameters (alpha, gamma, and beta)
    vr = np.sqrt(mu_sun/r)*((1 - (beta*np.cos(alpha)**2)*(np.cos(alpha)-np.tan(gamma)*np.sin(alpha)))**0.5)*np.sin(gamma)
    vtheta = np.sqrt(mu_sun/r)*((1 - (beta*np.cos(alpha)**2)*(np.cos(alpha)-np.tan(gamma)*np.sin(alpha)))**0.5)*np.cos(gamma)

    r_dot = vr
    theta_dot = vtheta/r

    return [r_dot,
            theta_dot,
            ]

t = np.linspace(0, 2000*24*60*60/tu, 200000)


def Marr(t, y):  # Mars arrival event to stop ode solver
    return y[0] - 228e6/du

Marr.direction = 1
Marr.terminal = True

sol = solve_ivp(dSdt, (0, 2000*24*60*60/tu), y0=[r0, theta0], method='RK45', t_eval=t, events=Marr)

t = sol.t
r = sol.y[0]
theta = sol.y[1]
x = []; y = []


for i in range(len(t)):

    r_p = r.item(i)
    theta_p = theta.item(i)

    x_temp = r_p * np.cos(theta_p)
    y_temp = r_p * np.sin(theta_p)
    x.append(x_temp)
    y.append(y_temp)

# Circles and on-plot text must be rearranged for each configuration

print("Transfer time = ", t[-1]*tu/60/60/24, "days")
x = np.array(x)
y = np.array(y)
fig, ax = plt.subplots()
circleE = plt.Circle((x[0]*du/AU, y[0]), 0.1, color='b')
circleS = plt.Circle((0, 0), 0.2, color='y')
circleM = plt.Circle((x[-1]*du/AU, y[-1]*du/AU), 0.05, color='r')
ax.add_patch(circleE)
ax.add_patch(circleS)
ax.add_patch(circleM)
ax.set_aspect('equal', adjustable='datalim')
plt.text(0.8, -0.2, 'Earth')
plt.text(-0.1, -0.3, 'Sun')
plt.text(1.45, 0.6, 'Mars')
plt.plot(x*du/AU, y*du/AU)
plt.ylabel('y (AU)', fontsize=22)
plt.xlabel('X (AU)', fontsize=22)
#plt.title('2D - Log. Spiral Trajectory', fontsize=22)
plt.show()
