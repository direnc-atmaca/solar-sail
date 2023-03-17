
'''
Author: Direnc Atmaca
Date of comment: March 17, 2023

Purpose: This code simulates the Earth Escape Trajectory for the solar sail
'''

# Importing necessary packages for numerical simulations, math, and plotting
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp


tu = 806.8; du = 6378.136  # canonical units set in a way to lead mu = 1

AU = 149597870.691  # 1 AU or the distance between the Earth and Sun in km
eps = np.radians(23.45)  # ecliptic tilt

# Gravitational parameters of the Sun and the Moon in canonical units - Not necessary for the current model
# mu_sun = (1.327124*(10**11))*((tu**2)/(du**3))
# mu_moon = (4.9028695*(10**3))*((tu**2)/(du**3))

J2 = 1.082e-3  # J2 perturbation nondimensional constant
Cd = 2.2  # Drag coefficient

Re = 6378.136/du  # Radius of the Earth
mu = 1  # Gravitational parameter of the Earth in canonical unit
we = (2*np.pi)/(86164/tu)  # Rotation rate of the Earth around its axis

# Initial orbit elements
p0 = 7138/du; e0 = 0; i0 = math.radians(90); raan0 = 0
om0 = 0; true_an0 = 0

# Initial orbit elements in terms of Modified Equinoctial elements
x1i = p0
x2i = e0*np.cos(raan0+om0)
x3i = e0*np.sin(raan0+om0)
x4i = np.tan(i0/2)*np.cos(raan0)
x5i = np.tan(i0/2)*np.sin(raan0)
x6i = raan0 + om0 + true_an0
x7i = 1
th_s_i = 0

clock = np.radians(90)  # clock angle

'''
Parameters below are the surface area (A) and characteristic acceleration (k) for various lightness numbers
To run the simulation appropriate ones must be uncommented while all others remain commented
'''
# Parameters of the spacecraft
m = 500  # kg

# lightness number = 0.015
#A = (4901.96e-6)/(du**2)
#k = (8.94e-8)*0.85*(tu**2)/du #beta = 0.015

# lightness number = 0.02
#A = (6535.95e-6)/(du**2)
#k = (1.1922e-7)*0.85*(tu**2)/du

# lightness number = 0.05
#A = (16339.87e-6)/(du**2)
#k = (2.98e-7)*0.85*(tu**2)/du

# lightness number = 0.1
#A = (32679.74e-6)/(du**2)
#k = (5.96e-7)*0.85*(tu**2)/du

# lightness number = 0.125
#A = (39215.69e-6)/(du**2)
#k = (7.15e-7)*0.85*(tu**2)/du

# lightness number = 0.15
A = (49019.61e-6)/(du**2)
k = (8.94e-7)*0.85*(tu**2)/du

# Atmospheric density function for the Earth atmosphere, automatically calculates density in kg per cubic meter
def atm_dens(z):
    c = 0
    h = np.array([0, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
         150, 180, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000])

    r = np.array([1.225, 4.008e-2, 1.841e-2, 3.996e-3, 1.027e-3, 3.097e-4, 8.283e-5,
         1.846e-5, 3.416e-6,5.606e-7, 9.708e-8, 2.222e-8, 8.152e-9, 3.831e-9,
         2.076e-9, 5.194e-10, 2.541e-10, 6.073e-11, 1.916e-11, 7.014e-12, 2.803e-12,
         1.184e-12, 5.215e-13, 1.137e-13, 3.070e-14, 1.136e-14, 5.759e-15, 3.561e-15])

    H = np.array([7.310, 6.427, 6.546, 7.360, 8.342, 7.583, 6.661,
         5.927, 5.533, 5.703, 6.782, 9.973, 13.243, 16.322,
         21.652, 27.974, 34.934, 43.342, 49.755, 54.513, 58.019,
         60.980, 65.654, 76.377, 100.587, 147.203, 208.020])

    if z > 1000:
        z = 1000
    elif z < 0:
        z = 0

    for j in range(0, 26):
        if h.item(j) <= z < h.item(j + 1):
            c = j

    if z == 1000:
        c = 26

    density = r.item(c)*np.exp(-(z - h.item(c))/H.item(c))
    return density

# Rotation matrix to convert Earth Centered Inertial Coord. to Local-Vertical Local-Horizontal (LVLH)
def ECItoLVLH(theta_t, i, raan):

    rot_mat = np.matrix([[np.cos(theta_t)*np.cos(raan)-np.sin(theta_t)*np.cos(i)*np.sin(raan), np.cos(theta_t)*np.sin(raan)+np.sin(theta_t)*np.cos(i)*np.cos(raan), np.sin(theta_t)*np.sin(i)],
                     [-np.sin(theta_t)*np.cos(raan)-np.cos(theta_t)*np.cos(i)*np.sin(raan), -np.sin(theta_t)*np.sin(raan)+np.cos(theta_t)*np.cos(i)*np.cos(raan), np.cos(theta_t)*np.sin(i)],
                     [np.sin(i)*np.sin(raan), -np.sin(i)*np.cos(raan), np.cos(i)]])
    return rot_mat

# Rotation matrix to convert Modified equinoctial elements (MEE) to ECI
def convtoECI(x1, x4, x5, x6, nu):
    x = (x1/nu)*(((x4**2-x5**2 + 1)*np.cos(x6) + 2*x4*x5*np.sin(x6))/(x4**2+x5**2 + 1))
    y = (x1/nu)*(((x5**2-x4**2 + 1)*np.sin(x6) + 2*x4*x5*np.cos(x6))/(x4**2+x5**2 + 1))
    z = (2*x1/nu)*((x4*np.sin(x6) - x5*np.cos(x6))/(x4**2+x5**2 + 1))
    coord = np.array([x, y, z])
    return coord


# Differential System
def dSdt(t, S):
    x1, x2, x3, x4, x5, x6, th_s = S

    print(t)  # track time to see simulation progress
    i = 2*np.arctan(np.sqrt(x4**2 + x5**2))
    raan = np.arctan2(x5, x4)
    theta_t = x6 - raan
    e = np.sqrt(x2**2 + x3**2)

    w = 0
    true_an = theta_t - w
    zeta = np.arctan(np.cos(theta_t)*np.tan(i))

    R = x1/(x2*np.cos(x6) + x3*np.sin(x6) + 1)

    nu = 1 + x2*np.cos(x6) + x3*np.sin(x6)

    G = np.sqrt(x1/mu)* \
        np.matrix([[0, 2*x1/nu, 0], [np.sin(x6), ((nu+1)*np.cos(x6)+x2)/nu, -((x4*np.sin(x6)-x5*np.cos(x6))/nu)*x3],
                   [-np.cos(x6), ((nu+1)*np.sin(x6)+x3)/nu, ((x4*np.sin(x6)-x5*np.cos(x6))/nu)*x2],
                   [0, 0, (1 + x4**2 + x5**2)*np.cos(x6)/(2*nu)], [0, 0, (1 + x4**2 + x5**2)*np.sin(x6)/(2*nu)]])

    true_an = true_an % (2*np.pi)  # map true anomaly to 0-2pi range
    alpha = np.pi/4 + true_an/2

    # Steering Law implementation for quadrants
    if np.radians(90) < true_an < np.radians(270):
        alpha = alpha - true_an
    elif np.radians(270) <= true_an < np.radians(360):
        alpha = alpha - np.radians(180)

    # Earth's gravitational harmonics - J2 component
    aJ2_r = ((3*mu)/R**4)*(Re**2)*J2*((3*(np.sin(theta_t)**2)*(np.sin(i)**2) - 1)/2)
    aJ2_th = -((3*mu)/R**4)*(Re**2)*J2*(np.sin(i)**2)*np.sin(theta_t)*np.cos(theta_t)
    aJ2_h = -((3*mu)/R**4)*(Re**2)*J2*np.sin(i)*np.cos(i)*np.sin(theta_t)

    # Drag perturbation
    vr = np.sqrt(mu/x1)*e*np.sin(true_an)
    ve = np.sqrt(mu/x1)*(1 + e*np.cos(true_an))*np.cos(zeta)
    vn = np.sqrt(mu/x1)*(1 + e*np.cos(true_an))*np.sin(zeta)
    vR = np.array([vr, ve*np.cos(zeta)-we*R*np.cos(i)+vn*np.sin(zeta), -ve*np.sin(zeta)+we*R*np.cos(theta_t)*np.sin(i)+vn*np.cos(zeta)])
    h = (R-Re)*du
    rho = atm_dens(h)*10**(9)*(du**(3))
    aD = (-1/2)*Cd*(A/m)*rho*(np.linalg.norm(vR))*vR
    aD_r = aD.item(0); aD_th = aD.item(1); aD_h = aD.item(2)

    # Putting the perturbing acceleration components together
    aP_r = aJ2_r + aD_r
    aP_th = aJ2_th + aD_th
    aP_h = aJ2_h + aD_h
    aP = np.array([[aP_r], [aP_th], [aP_h]])

    # Thrust acceleration
    aT = k*(np.cos(alpha)**2)*np.array([np.cos(alpha), np.sin(alpha)*np.cos(clock), np.sin(alpha)*np.sin(clock)])

    a = aT + aP  # total acceleration

    z_dot = G*a

    x6_dot = np.sqrt(mu/x1**3)*(nu**2) + np.sqrt(x1/mu)*((x4*np.sin(x6)-x5*np.cos(x6))/nu)*a[2]
    x6_dot = x6_dot.item(0)
    x1_dot = z_dot.item(0); x2_dot = z_dot.item(1); x3_dot = z_dot.item(2); x4_dot = z_dot.item(3); x5_dot = z_dot.item(4)
    th_s_dot = (2*np.pi)/(365.25*24*60*60/tu)

    return [x1_dot,
            x2_dot,
            x3_dot,
            x4_dot,
            x5_dot,
            x6_dot,
            th_s_dot]


t = np.linspace(0, 3000*24*60*60/tu, 1000000)

# Escape trajectory event to stop simulation when the eccentricity exceeds 1
def EccStop(t, y):

    cond = np.sqrt(y[1]**2 + y[2]**2) - 0.95 #
    return cond


EccStop.direction = 1
EccStop.terminal = True

# Solver definition
sol = solve_ivp(dSdt, (0, 3000*24*60*60/tu), y0=[x1i, x2i, x3i, x4i, x5i, x6i, th_s_i], method='DOP853', t_eval=t, events = EccStop)

# Collect the results
t = sol.t
x1 = sol.y[0]
x2 = sol.y[1]
x3 = sol.y[2]
x4 = sol.y[3]
x5 = sol.y[4]
x6 = sol.y[5]
th_s = sol.y[6]

# Calculate orbit elements from the results
inc = 2*np.arctan(np.sqrt(x4**2 + x5**2))
e = np.sqrt(x2**2 + x3**2)

# Define necessary outputs
aTr = []; aTth = []; aTh = []; aT_tot = []
aD_tot = []
X = []; Y = []; Z = []
a = []

# Post-processing to generate the elementary parameters cannot be extracted during the simulation
for i in range(len(t)):

    nu = 1 + x2.item(i)*np.cos(x6.item(i)) + x3.item(i)*np.sin(x6.item(i))
    ECI = convtoECI(x1.item(i), x4.item(i), x5.item(i), x6.item(i), nu)
    x_temp = ECI.item(0); y_temp = ECI.item(1); z_temp = ECI.item(2)
    X.append(x_temp)
    Y.append(y_temp)
    Z.append(z_temp)
    ecc = np.sqrt(x2.item(i)**2 + x3.item(i)**2)
    a_temp = x1.item(i)/(1 - ecc**2)
    a.append(a_temp)

    raan = np.arctan2(x5.item(i), x4.item(i))
    theta_t = x6.item(i) - raan
    zeta = np.arctan(np.cos(theta_t)*np.tan(inc.item(i)))

    R = x1.item(i)/(x2.item(i)*np.cos(x6.item(i)) + x3.item(i)*np.sin(x6.item(i)) + 1)
    w = 0
    true_an = theta_t - w
    true_an = true_an % (2*np.pi)

    alpha = np.pi/4 + true_an/2
    if np.radians(90) < true_an < np.radians(270):
        alpha = alpha - true_an
    elif np.radians(270) <= true_an < np.radians(360):
        alpha = alpha - np.radians(180)

    aT = k*(np.cos(alpha)**2)*np.array([np.cos(alpha), np.sin(alpha)*np.cos(clock), np.sin(alpha)*np.sin(clock)])
    aTr_temp = aT.item(0); aTth_temp = aT.item(1); aTh_temp = aT.item(2)
    aT_tot_temp = np.sqrt(aTr_temp**2 + aTth_temp**2 + aTh_temp**2)
    aT_tot.append(aT_tot_temp)

    '''
    Drag perturbation elements were plotted for investigation, it can be omitted here
    vr = np.sqrt(mu/x1.item(i))*ecc*np.sin(true_an)
    ve = np.sqrt(mu/x1.item(i))*(1 + ecc*np.cos(true_an))*np.cos(zeta)
    vn = np.sqrt(mu/x1.item(i))*(1 + ecc*np.cos(true_an))*np.sin(zeta)
    vR = np.array([vr, ve*np.cos(zeta)-we*R*np.cos(inc.item(i))+vn*np.sin(zeta), -ve*np.sin(zeta)+we*R*np.cos(theta_t)*np.sin(inc.item(i))+vn*np.cos(zeta)])
    h = (R-Re)*du
    rho = atm_dens(h)*10**(9)*(du**(3))
    aD = (-1/2)*Cd*(A/m)*rho*(np.linalg.norm(vR))*vR
    aDr_temp = aD.item(0); aDth_temp = aD.item(1); aDh_temp = aD.item(2)
    aD_tot_temp = np.sqrt(aDr_temp**2 + aDth_temp**2 + aDh_temp**2)
    aD_tot.append(aD_tot_temp)
    '''

plt.plot(t*tu/(60*60*24), x1*du)
plt.ylabel('p (km)', fontsize=22)
plt.xlabel('time (day)', fontsize=22)
plt.title('Semilatus Rectum', fontsize=22)
plt.show()

a = np.array(a)
plt.plot(t*tu/(60*60*24), a*du)
plt.ylabel('a (km)', fontsize=22)
plt.xlabel('time (day)', fontsize=22)
plt.title('Semimajor axis', fontsize=22)
plt.show()

plt.plot(t*tu/(60*60*24), e)
plt.ylabel('e', fontsize=22)
plt.xlabel('time (day)', fontsize=22)
plt.title('Eccentricity', fontsize=22)
plt.show()

plt.plot(t*tu/(60*60*24), inc*180/math.pi)
plt.ylabel('i (degrees)', fontsize=22)
plt.xlabel('time (day)', fontsize=22)
plt.title('Inclination', fontsize=22)
plt.show()

aT_tot = np.array(aT_tot)
plt.plot(t*tu/(60*60*24), aT_tot*(du/(tu**2))*1e3)
plt.ylabel(r'$a_T$', fontsize=22)
plt.xlabel('time (day)', fontsize=22)
plt.title('Total Thrust Acceleration', fontsize=22)
plt.show()


print("Transfer time = ", t[-1]*tu/60/60/24, "days")
#print(type(X))
X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)
ax = plt.axes()
ax.axis('equal')
#ax = plt.axes(projection='3d')
#plt.plot(X*du, Y*du, Z*du)
plt.plot(X*du, Z*du)
ax.set_xlabel('x (km)')
ax.set_ylabel('z (km)')
#ax.set_zlabel('z (km)')
plt.show()


'''
aD_tot = np.array(aD_tot)
plt.plot(t*tu/(60*60*24), aD_tot*(du/(tu**2))*1e3)
plt.ylabel(r'$a_D$', fontsize=22)
plt.xlabel('time (day)', fontsize=22)
plt.title('Drag Perturbing Acceleration', fontsize=22)
plt.show()
'''
