
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')


# In[2]:

import scipy
import numpy as np #outils mathématiques de base
import matplotlib.pyplot as plt #dessins
import pandas as pd

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# In[3]:

from scipy.integrate import odeint


# In[4]:

periode = 100
nombre_points = 100000
t = np.linspace(0,periode,nombre_points) # 100 points temporels de 0 à 20

# Fonctions de comportement
#coefficients phi1 et phi2 de la courbe de Philips
# Emmanuel dit : \phi(\lambda) = -0,73502 + 1.08519*\lambda

phi1 = -0.04/(1-0.04**2)
phi2 = 0.04**3/(1-0.04**2)
def f (l): #courbe de Phillips
#	return -0.73502 + 1.08519*l
	return phi1 + phi2/(1-min(l,0.99))**2	#-0.73502 + 1.08519*l

# Emmanuel dit : \kappa(\pi) = 0.04260 + 0.64153*\pi
k_min = 0.05
k_max = 1
def k(p):
	return min(k_max, max(k_min, -0.0065 + np.exp(-5+20*p)))

# In[5]:

#Paramètres par défaut
"""
#Policy
tau = 0 #taxes
DeltapE = 0

# Taux de croissance
delta = 0.04 #taux de dépréciation, source Coping
alpha = 0.02 #accroissement de la productivité du travail, source Coping
beta = 0.02 #accroissement de la population
dPE = 0 #1 #taux de croissance du prix de l'énergie


#Autres
r = 0.03 #taux d'intérêt, même hypothèse que Coping
nu1 = 1/2.7 #productivité de K1, source Coping
nu2 = nu1 - 0.05 #productivité de K2
sigma = 1.0 #arbitraire

#Valeurs initiales
w0 = 0.58496 #source Coping
l0 = 0.691 #source Coping
d0 = 1.43931 #source Coping
eps0 = 0.9
th0 = 0.5
pE0 = 0.08
Y0=1
Em0 = 0 #émissions cumulées
"""

#Policy
tau = 0 #taxes
DeltapE = 0

# Taux de croissance
delta = 0.04 #0.01 #taux de dépréciation
alpha = 0.02 #accroissement de la productivité du travail
beta = 0.02 #accroissement de la population
dPE = 0 #1 #taux de croissance du prix de l'énergie


#Autres
r = 0.03 #taux d'intérêt
nu1 = 1/2.7 #productivité de K1
nu2 = nu1-0.05 #productivité de K2
sigma = 1.0

#Valeurs initiales
w0 = 0.5
l0 = 0.8
d0 = 0.0
eps0 = 0.9
th0 = 0.5
pE0 = (nu1-nu2)+0.03
Y0=1
Em0 = 0 #émissions cumulées

# In[6]:

#Fonctions usuelles
def pE(t):
	return pE0 #if t < 70 else pE1 #pE0*np.exp(dPE*t)

def inv_pub(y,t): # public investment
	[l,eps] = [y[1],y[3]]
	return tau+DeltapE*eps/nu1 if l < 1 else 0 #balanced budget condition / 0 public investment in full employment situation

def pr(y,t): #profit
	[w,l,d,eps,th] = y[0:5]
	return 1 - w - r*d - (pE(t)-DeltapE)*eps/nu1 - inv_pub(y,t)

def thG(th): #public investment allocation as a function of private theta
	return th #default : public allocation follows private

def nu_th(y,t):
	[w,l,d,eps,th] = y[0:5]
	return nu1 - (nu1 - nu2)*(th*inv_prive(y,t)+thG(th)*inv_pub(y,t))/(inv_prive(y,t)+inv_pub(y,t))
	# (theta bar est la moyenne pondérée du theta public et du theta privé)

def mu_eps(eps):
	return nu1*(1-eps)+nu2*eps

def rho(y,t): #rate of profit
	[w,l,d,eps,th] = y[0:5]
	return nu1*nu2*pr(y,t)/mu_eps(eps)

def inv_prive(y,t): #private investment to output ratio
	[w,l,d,eps,th] = y[0:5]
	inv1 = k(rho(y,t)/nu1)/(rho(y,t)/nu1)*pr(y,t)
	if l<1:
		return inv1
	else: #investment constrained by full employment (assuming zero public inv)
		nu_bar = nu1*nu2/mu_eps(eps) #Average productivity of capital
		return min(inv1,(alpha + beta + delta)/nu_bar)

def inv(y,t): #total investment to output ratio
	return inv_prive(y,t) + inv_pub(y,t)


def lpoint(y,t):
	[w,l,d,eps,th] = y[0:5]
	lp1 = l*((nu1*nu2/mu_eps(eps))*inv(y,t) - (alpha+beta+delta))
	return lp1 if (l < 1 or lp1 < 0) else 0 #full employment ceiling

def epspoint(y,t):
	[w,l,d,eps,th] = y[0:5]
	return (nu_th(y,t)*mu_eps(eps)-nu1*nu2)*inv(y,t)/(nu1-nu2)
	
def wpoint(y,t):
	[w,l,d,eps,th] = y[0:5]
	wp1 = w*(f(l)-(nu_th(y,t) - nu1*nu2/mu_eps(eps))*inv(y,t)-alpha)
	#return w*((1-w)*f(l)-(nu_th(y,t) - nu1*nu2/mu_eps(eps))*inv(y,t)-alpha)
	return wp1 if w < 1 or wp1 < 0 else 0

# In[7]:

def differentielle (y, t):
	[w,l,d,eps,th,Y] = y[0:6]
	return [wpoint(y,t),
			lpoint(y,t),
			inv_prive(y,t)-pr(y,t)-d*(nu_th(y,t)*inv(y,t)-delta),
			epspoint(y,t),
			sigma*th*(1-th)*(pE(t)-(nu1-nu2)),
			(nu_th(y,t)*inv(y,t)-delta)*Y,
			eps*Y/nu1
		   ]
