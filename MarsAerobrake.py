
'''
Author: Direnc Atmaca
Date of comment: March 17, 2023

Purpose: This code simulates the Aerobraking trajectory at Mars
'''

# Importing necessary packages for numerical simulations, math, and plotting
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
plt.rcParams['agg.path.chunksize'] = 10000  # this line increases the number of points allowed for plotting

tu = 956.37; du = 3396.2 # canonical units for Mars
Cd = 2.2  # Drag coefficient

Rm = 3396.2/du  # Radius of the Mars
mu = 1  # Gravitational parameter of the Mars in canonical units

wm = (2*np.pi)/(88642/tu)  # rotation rate of Mars around its own axis

# Mars Capture Orbit definitions
ra = 45000 + du
rp = 225 + du
a = ((ra+rp)/2)

# Initial Orbit Elements
e0 = (ra-rp)/(ra+rp)
p0 = (a*(1 - e0**2))/du
true_an0 = 0

'''
Parameters below are the surface area (A), lightness number (beta), and mass (m)
To run the simulation appropriate beta and mass must be uncommented while all others remain commented
'''

# Parameters of the Sail
A = (6535.95e-6)/(du**2)  # area, m2

# Mass is scaled for a constant surface area, this is equivalent to changing the surface area for a constant mass

beta = 0.015
m = 666.67  # kg

#beta = 0.02
#m = 500  # kg

#beta = 0.05
#m = 200  # kg

#beta = 0.1
#m = 100  # kg

#beta = 0.125
#m = 80  # kg

#beta = 0.15
#m = 66.67  # kg

# Atmospheric density model for the Mars
def atm_dens(z):

    z = z*1000  # km-to-meter conversion

    if z > 7000:
        T = -23.4 - 0.00222*z  # temperature
    else:
        T = -31 - 0.000998*z

    P = 0.699*np.exp(-0.00009*z)  # pressure

    density = P/(0.1921*(T+273.1))  # kg per cubic meter
    return density

# Differential System
def dSdt(t, S):

    p, e, true_an = S

    R = p/(1 + e*np.cos(true_an))

    vr = np.sqrt(mu/p)*e*np.sin(true_an)
    ve = np.sqrt(mu/p)*(1 + e*np.cos(true_an))

    vR = np.array([vr, ve - wm*R])
    h = (R-Rm)*du
    rho = atm_dens(h)*10**(9)*(du**(3))
    aD = (1/2)*Cd*(A/m)*rho*(np.linalg.norm(vR))*vR
    a_r = aD.item(0); a_th = aD.item(1)

    p_dot = 2*np.sqrt(p/mu)*R*a_th
    e_dot = np.sqrt(p/mu)*a_r*np.sin(true_an) + np.sqrt(p/mu)*a_th*((e + e*np.cos(true_an)**2 + 2*np.cos(true_an))/(1 + e*np.cos(true_an)))
    true_an_dot = np.sqrt(mu/(p**3))*((1 + e*np.cos(true_an))**2) + (a_r*np.cos(true_an)/e)*np.sqrt(p/mu) \
                  - a_th*np.sqrt(p/mu)*np.sin(true_an)*((e*np.cos(true_an) + 2)/(e*(1 + e*np.cos(true_an))))

    return [p_dot,
            e_dot,
            true_an_dot,
            ]


t = np.linspace(0, 1000*24*60*60/tu, 2000000)


def LEOMars(t, y):  # Event to stop propagation once the apogee is in low Mars orbit
    p_ev = y[0]
    e_ev = y[1]
    a_ev = p_ev/(1 - e_ev**2)
    ra_ev = a_ev*(1 + e_ev)
    return ra_ev - (300 + du)/du


LEOMars.direction = 1
LEOMars.terminal = True

sol = solve_ivp(dSdt, (0, 1000*24*60*60/tu), y0=[p0, e0, true_an0], method='RK45', t_eval=t, events=LEOMars, rtol=1e-12, atol=1e-12)

t = sol.t
p = sol.y[0]
e = sol.y[1]
true_an = sol.y[2]

x = []; y = []
aD_tot = []


for i in range(len(t)):
    print(i)
    R = p.item(i)/(1 + e.item(i)*np.cos(true_an.item(i)))

    '''
    vr = np.sqrt(mu/p.item(i))*e.item(i)*np.sin(true_an.item(i))
    ve = np.sqrt(mu/p.item(i))*(1 + e.item(i)*np.cos(true_an.item(i)))

    vR = np.array([vr, ve - wm*R])

    h = (R-Rm)*du
    rho = atm_dens(h)*10**(9)*(du**(3))
    aD = (-1/2)*Cd*(A/m)*rho*(np.linalg.norm(vR))*vR
    aDr_temp = aD.item(0); aDth_temp = aD.item(1)
    aD_temp = np.sqrt(aDr_temp**2 + aDth_temp**2)
    aD_tot.append(aD_temp)
    '''
    x_temp = R * np.cos(true_an.item(i))
    y_temp = R * np.sin(true_an.item(i))
    x.append(x_temp)
    y.append(y_temp)


print("Transfer time = ", t[-1]*tu/60/60/24, "days")
x = np.array(x)
y = np.array(y)
fig, ax = plt.subplots()
circleM = plt.Circle((0, 0), 3396.2, color='r')
ax.add_patch(circleM)
plt.plot(x*du, y*du)
plt.ylabel('y', fontsize=22)
plt.xlabel('x', fontsize=22)
#plt.title('Mars - Aerobraking ', fontsize=22)
plt.show()

'''
aD_tot = np.array(aD_tot)
plt.plot(t*tu/(60*60*24), aD_tot*(du/(tu**2))*1e3)
plt.ylabel(r'$a_D$', fontsize=22)
plt.xlabel('time (day)', fontsize=22)
plt.title('Drag Perturbing Acceleration', fontsize=22)
plt.show()
'''
