
# SUPPPLMENTARY FUNCTIONS 

import SUAVE 
from SUAVE.Core import Data , Units  
import math 
import operator as op 
import matplotlib.pyplot as plt 
import numpy as np    
from scipy.interpolate import interp1d,  CubicSpline , UnivariateSpline,  PPoly  
from SUAVE.Plots.Geometry import plot_airfoil 
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_geometry\
     import import_airfoil_geometry
from SUAVE.Methods.Geometry.Two_Dimensional.Cross_Section.Airfoil.import_airfoil_polars \
     import import_airfoil_polars
from SUAVE.Plots.Performance.Airfoil_Plots import *
import os     


from MAIN import * 
from structures import * 
from geometry import * 


# =========================================================
# paneling_functions
# =========================================================    

#-------------------------------------------------------------------------------
def make_panels(M, npanel):
    # places panels on the current airfoil, as described by M.geom.xpoint
    # INPUT
    #   M      : mfoil class
    #   npanel : number of panels
    # OUTPUT
    #   M.foil.N : number of panel points
    #   M.foil.x : coordinates of panel nodes (2xN)
    #   M.foil.s : arclength values at nodes (1xN)
    #   M.foil.t : tangent vectors, not np.linalg.normalized, dx/ds, dy/ds (2xN)
    # DETAILS
    #   Uses curvature-based point distribution on a spline of the points
 
    Ufac  = 2  # uniformity factor (higher, more uniform paneling)
    TEfac = 0.1 # Trailing-edge factor (higher, more TE resolution)
    M.foil.x, M.foil.s, M.foil.t = spline_curvature(M.geom.xpoint, npanel+1, Ufac, TEfac)
    M.foil.N = len(M.foil.x[1])

    return 

#-------------------------------------------------------------------------------
def TE_info(X):
    # returns trailing-edge information for an airfoil with node coords X
    # INPUT
    #   X : node coordinates, ordered clockwise (2xN)
    # OUTPUT
    #   t    : bisector vector = average of upper/lower tangents, np.linalg.normalized
    #   hTE  : trailing edge gap, measured as a cross-section
    #   dtdx : thickness slope = d(thickness)/d(wake x)
    #   tcp  : |t cross p|, used for setting TE source panel strength
    #   tdp  : t dot p, used for setting TE vortex panel strength
    # DETAILS
    #   p refers to the unit vector along the TE panel (from lower to upper)

    t1   = X[:, 0]-X[:,1]
    t1   = t1/np.linalg.norm(t1) # lower tangent vector
    t2   = X[:,-1]-X[:,-2] 
    t2   = t2/np.linalg.norm(t2) # upper tangent vector
    t    = 0.5*(t1+t2) 
    t    = t/np.linalg.norm(t) # average tangent gap bisector
    s    = X[:,-1]-X[:,0] # lower to upper connector vector
    hTE  = -s[0]*t[1] + s[1]*t[0] # TE gap
    dtdx = t1[0]*t2[1] - t2[0]*t1[1] # sin(theta between t1,t2) approx dt/dx
    p    = s/np.linalg.norm(s) # unit vector along TE panel
    tcp  = abs(t[0]*p[1]-t[1]*p[0]) 
    tdp  = np.dot(t,p)

    return t, hTE, dtdx, tcp, tdp


#-------------------------------------------------------------------------------
def panel_linvortex_stream(Xj, xi):
    # calculates the streamdef coefficients for a linear vortex panel
    # INPUTS
    #   Xj  : X(:,[1,2]) = panel point coordinates
    #   xi  : control point coordinates (2x1)
    # OUTPUTS
    #   a,b : streamdef influence coefficients
    # DETAILS
    #   The streamdef due to the panel is then a*g1 + b*g2
    #   where g1 and g2 are the vortex strengths at the panel points

    # panel coordinates
    xj1 = Xj[0,0]  
    zj1 = Xj[1,0]
    xj2 = Xj[0,1]  
    zj2 = Xj[1,1]

    # panel-aligned tangent and np.linalg.normal vectors
    t = np.array([[xj2-xj1 ],[ zj2-zj1]])
    t = t/np.linalg.norm(t)
    n = np.array([-t[1], t[0]])

    # control point relative to (xj1,zj1)
    xz = np.array([(xi[0]-xj1), (xi[1]-zj1)]).T
    x = np.dot(xz,t) # in panel-aligned coord system
    z = np.dot(xz,n)  # in panel-aligned coord system

    # distances and angles
    d      = np.linalg.norm([xj2-xj1, zj2-zj1]) # panel length
    r1     = np.linalg.norm([x,z])             # left edge to control point
    r2     = np.linalg.norm([x-d,z])           # right edge to control point
    theta1 = math.atan2(z,x)          # left angle
    theta2 = math.atan2(z,x-d)        # right angle

    # check for r1, r2 zero
    ep = 1e-10
    if (r1 < ep):
        logr1 =  np.array([0]) 
    else:
        logr1 = np.log(r1) 
    if (r2 < ep):
        logr2 = np.array([0]) 
    else:
        logr2 = np.log(r2) 

    # streamdef components
    P1 = (0.5/np.pi)*(z*(theta2-theta1) - d + x*logr1 - (x-d)*logr2)
    P2 = x*P1 + (0.5/np.pi)*(0.5*r2**2*logr2 - 0.5*r1**2*logr1 - r2**2/4 + r1**2/4)

    # influence coefficients
    a = P1-P2/d
    b =    P2/d

    return a, b


#-------------------------------------------------------------------------------
def panel_constsource_stream(Xj, xi):
    # calculates the streamdef coefficient for a constant source panel
    # INPUTS
    #   Xj    : X(:,[1,2]) = panel point coordinates
    #   xi    : control point coordinates (2x1)
    # OUTPUTS
    #   a     : streamdef influence coefficient of the panel
    # DETAILS
    #   The streamdef due to the panel is then a*s
    #   where s is the panel source strength

    # panel coordinates
    xj1 = Xj[0,0]  
    zj1 = Xj[1,0]
    xj2 = Xj[0,1]  
    zj2 = Xj[1,1]

    # panel-aligned tangent and np.linalg.normal vectors
    t = np.array([xj2-xj1, zj2-zj1 ])
    t = t/np.linalg.norm(t)
    n = np.array([-t[1],t[0]])

    # control point relative to (xj1,zj1)
    xz = np.array([(xi[0]-xj1), (xi[1]-zj1)])
    x = np.dot(xz,t) # in panel-aligned coord system
    z = np.dot(xz,n)  # in panel-aligned coord system        

    # distances and angles
    d = np.linalg.norm([xj2-xj1, zj2-zj1]) # panel length
    r1 = np.linalg.norm([x,z])             # left edge to control point
    r2 = np.linalg.norm([x-d,z])           # right edge to control point

    theta1 = math.atan2(z,x)          # left angle
    theta2 = math.atan2(z,x-d)        # right angle

    # streamdef
    ep = 1e-9
    if (r1 < ep): 
        logr1  = np.array([0]) 
        theta1 = np.array([np.pi]) 
        theta2 = np.array([np.pi]) 
    else: 
        logr1 = np.log(r1) 
    if (r2 < ep): 
        logr2  = np.array([0]) 
        theta1 = np.array([0]) 
        theta2 = np.array([0]) 
    else: 
        logr2 = np.log(r2) 
    P = (x*(theta1-theta2) + d*theta2 + z*logr1 - z*logr2)/(2*np.pi)

    dP = d # delta psi
    if ((theta1+theta2) > np.pi):
        P = P - 0.25*dP 
    else:
        P = P + 0.75*dP
    

    # influence coefficient
    a = P
     
    return a 
