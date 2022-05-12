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
from supporting_functions import * 

# =========================================================
# INVISCID FINCTIONS 
# =========================================================  
def solve_inviscid(M):
    # solves the inviscid system, rebuilds 0,90deg solutions
    # INPUT
    #   M : mfoil structure
    # OUTPUT
    #   inviscid vorticity distribution is computed
    # DETAILS
    #   Uses the angle of attack in M.oper.gamma
    #   Also initializes thermo variables for normalization
    if M.foil.N>0: 
        assert('No panels')
    M.oper.viscous = False
    init_thermo(M)
    M.isol.sgnue = np.ones((1,M.foil.N)) # do not distinguish sign of ue if inviscid
    build_gamma(M, M.oper.alpha)
    if (M.oper.givencl):
        cltrim_inviscid(M) 
    calc_force(M)
    M.glob.conv = True # no coupled system ... convergence is guaranteed
    return 

def cltrim_inviscid(M):
    # trims inviscid solution to prescribed target cl, using alpha
    # INPUT
    #   M : mfoil structure
    # OUTPUT
    #   inviscid vorticity distribution is computed for a given cl
    # DETAILS
    #   Iterates using cl_alpha computed in post-processing
    #   Accounts for cl_ue in total derivative

    for i in range(15): # trim iterations
        alpha = M.oper.alpha # current angle of attack
        calc_force(M) # calculate current cl and linearization
        R = M.post.cl - M.oper.cltgt
        if (np.linalg.norm(R) < 1E-10):  
            break 
        sc  = np.zeros(2,len(alpha))
        sc[0]= -np.sin(alpha)*np.pi/180
        sc[1]= np.cos(alpha)*np.pi/180 
        cl_a = M.post.cl_alpha + M.post.cl_ue*(M.isol.gamref*sc) # total deriv
        dalpha = -R/cl_a
        M.oper.alpha = alpha + min(max(dalpha,-2), 2)
    
    if (i>=15):
        print('** inviscid cl trim not converged **') 
        

    gam   = np.zeros(2,len(alpha))
    gam[0]= np.cos(M.oper.alpha)
    gam[1]= np.sin(M.oper.alpha) 
    M.isol.gam = M.isol.gamref*gam
    
    return 


def get_ueinv(M):
    # computes invicid tangential velocity at every node
    # INPUT
    #   M : mfoil structure
    # OUTPUT
    #   ueinv : inviscid velocity at airfoil and wake (if exists) points
    # DETAILS
    #   The airfoil velocity is computed directly from gamma
    #   The tangential velocity is measured + in the streamwise direction
    if np.shape(M.isol.gam) != 0:
        print('No inviscid solution')
    alpha = M.oper.alpha  
    cs  = np.zeros((2,len(alpha)))
    cs[0]= np.cos(alpha)
    cs[1]= np.sin(alpha) 
    uea   = (M.isol.sgnue*(np.matmul(M.isol.gamref,cs)).T ).T # airfoil
    if (M.oper.viscous) and (M.wake.N > 0):
        uew    = np.matmul(M.isol.uewiref,cs) # wake
        uew[0] = uea[-1] # ensures continuity of upper surface and wake ue
    else:
        uew = np.empty(shape=[0,1])
     
    ueinv= np.concatenate((uea, uew), axis = 0)# airfoil/wake edge velocity

    return ueinv

    
#-------------------------------------------------------------------------------
def get_ueinvref(M):
    # computes 0,90deg inviscid tangential velocities at every node
    # INPUT
    #   M : mfoil structure
    # OUTPUT
    #   ueinvref : 0,90 inviscid tangential velocity at all points (N+Nw)x2
    # DETAILS
    #   Uses gamref for the airfoil, uewiref for the wake (if exists)
    
    if np.shape(M.isol.gam) != 0:
        print('No inviscid solution')
    
    uearef = M.isol.sgnue.T*M.isol.gamref # airfoil
    if (M.oper.viscous) and (M.wake.N > 0):
        uewref = M.isol.uewiref # wake
        uewref[0,:] = uearef[-1,:] # continuity of upper surface and wake
    else:
        uewref = []  
    
    
    ueinvref = np.zeros((2,len(uearef)))
    ueinvref[0]= uearef
    ueinvref[1]= uewref    

    return ueinvref


#-------------------------------------------------------------------------------
def build_gamma(M, alpha):
    # builds and solves the inviscid linear system for alpha=0,90,input
    # INPUT
    #   M     : mfoil structure
    #   alpha : angle of attack (degrees)
    # OUTPUT
    #   M.isol.gamref : 0,90deg vorticity distributions at each node (Nx2)
    #   M.isol.gam    : gamma for the particular input angle, alpha
    #   M.isol.AIC    : aerodynamic influence coefficient matrix, filled in
    # DETAILS
    #   Uses streamdef approach: constant psi at each node
    #   Continuous linear vorticity distribution on the airfoil panels
    #   Enforces the Kutta condition at the TE
    #   Accounts for TE gap through const source/vorticity panels
    #   Accounts for sharp TE through gamma extrapolation

    N               = M.foil.N         # number of points  
    A               = np.zeros((N+1,N+1))  # influence matrix
    rhs             = np.zeros((N+1,2))  # right-hand sides for 0,90
    _,hTE,_,tcp,tdp = TE_info(M.foil.x) # trailing-edge info
    nogap = (abs(hTE) < 1e-10*M.geom.chord) # indicates no TE gap

    print('\n <<< Solving inviscid problem >>> \n')

    # Build influence matrix and rhs
    for i in range(N):             # loop over nodes
        xi = M.foil.x[:,i] # coord of node i
        for j in range(N-1):         # loop over panels
            [aij, bij] = panel_linvortex_stream(M.foil.x[:,[j,j+1]], xi)
            A[i,j  ]   = A[i,j  ] + aij
            A[i,j+1]   = A[i,j+1] + bij
            A[i,N]   = -1 # last unknown = streamdef value on surf
        
        # right-hand sides
        rhs[i,:] = [-xi[1], xi[0]]
        # TE source influence
        a = panel_constsource_stream(M.foil.x[:,[-1,0]], xi)
        A[i,1] = A[i,1] - a*(0.5*tcp)
        A[i,-2] = A[i,-2] + a*(0.5*tcp)
        # TE vortex panel
        [a, b] = panel_linvortex_stream(M.foil.x[:,[-1,0]], xi)
        A[i,1] = A[i,1] - (a+b)*(-0.5*tdp)
        A[i,-2] = A[i,-2] + (a+b)*(-0.5*tdp)
    

    # special Nth equation (extrapolation of gamma differences) if no gap
    if (nogap):
        A[-2,:]                = 0 
        A[-2,[0,1,2,-4,-3,-2]]= [1,-2,1,-1,2,-1] 

    # Kutta condition
    A[-1,0] = 1
    A[-1,-2] = 1

    # Solve system for unknown vortex strengths
    M.isol.AIC    = A
    g             = np.linalg.solve(M.isol.AIC,rhs)
    M.isol.gamref = g[:-1,:] # last value is surf streamdef
    M.isol.gam    = M.isol.gamref[:,0]*np.cos(alpha) + M.isol.gamref[:,1]*np.sin(alpha)

    return 


#-------------------------------------------------------------------------------
def inviscid_velocity(X, G, Vinf, alpha, x,nargout=1):
    # returns inviscid velocity at x due to gamma (G) on panels X, and Vinf
    # INPUT
    #   X     : coordinates of N panel nodes (N-1 panels) (Nx2)
    #   G     : vector of gamma values at each airfoil node (Nx1)
    #   Vinf  : freestream speed magnitude
    #   alpha : angle of attack (degrees)
    #   x     : location of point at which velocity vector is desired  
    # OUTPUT
    #   V    : velocity at the desired point (2x1)
    #   V_G  : (optional) linearization of V w.r.t. G, (2xN)
    # DETAILS
    #   Uses linear vortex panels on the airfoil
    #   Accounts for TE const source/vortex panel
    #   Includes the freestream contribution

    N     = len(X[1])   # number of points  
    V     = np.zeros(2)  # velocity
    dolin = False
    if (nargout > 1):
        dolin = True  # (nargout > 1) # linearization requested
    if (dolin):
        V_G = np.zeros((2,N))
    _,_,_,tcp,tdp  = TE_info(X) # trailing-edge info
    
    # assume x is not a midpoint of a panel (can check for this)
    for j in range(N-1): # loop over panels
        a, b = panel_linvortex_velocity(X[:,[j,j+1]], x, np.empty([0]), False)
        V = V + a*G[j] + b*G[j+1]
        if (dolin):
            V_G[:,j]  = V_G[:,j] + a 
            V_G[:,j+1] = V_G[:,j+1] + b 
    
    # TE source influence
    a  = panel_constsource_velocity(X[:,[-1,0]], x, np.empty([0]))
    f1 = a*(-0.5*tcp) 
    f2 = a*0.5*tcp
    V  = V + f1*G[0] + f2*G[-1]
    if (dolin):
        V_G[:,0] = V_G[:,0] + f1 
        V_G[:,-1] = V_G[:,-1] + f2 
        
    # TE vortex influence
    a,b = panel_linvortex_velocity(X[:,[-1,0]], x,np.empty([0]), False)
    f1  = (a+b)*(0.5*tdp) 
    f2  = (a+b)*(-0.5*tdp)
    V   = V + f1*G[0] + f2*G[-1]
    if (dolin):
        V_G[:,0] = V_G[:,0] + f1 
        V_G[:,-1] = V_G[:,-1] + f2 
        
    # freestream influence
    alf    = np.zeros(2)
    alf[0] = np.cos(alpha)
    alf[1] = np.sin(alpha)
    V      = V+ Vinf*alf

    if dolin:
        return V_G  
    else: 
        return V  
    



#-------------------------------------------------------------------------------
def build_wake(M):
    # builds wake panels from the inviscid solution
    # INPUT
    #   M     : mfoil class with a valid inviscid solution (gam)
    # OUTPUT
    #   M.wake.N  : Nw, the number of wake points
    #   M.wake.x  : coordinates of the wake points (2xNw)
    #   M.wake.s  : s-values of wake points (continuation of airfoil) (1xNw)
    #   M.wake.t  : tangent vectors at wake points (2xNw)
    # DETAILS
    #   Constructs the wake path through repeated calls to inviscid_velocity
    #   Uses a predictor-corrector method
    #   Point spacing is geometric prescribed wake length and number of points


    if np.shape(M.isol.gam) != 0:
        print('No inviscid solution to rebuild') 
    N    = M.foil.N  # number of points on the airfoil
    Vinf = M.oper.Vinf    # freestream speed
    Nw   = int(np.ceil(N/10 + 10*M.geom.wakelen)) # number of points on wake
    S    = M.foil.s  # airfoil S values
    ds1  = 0.5*(S[1]-S[0] + S[-1]-S[-2]) # first nominal wake panel size
    sv   = space_geom(ds1, M.geom.wakelen*M.geom.chord, Nw) # geometrically-spaced points
    xyw  = np.zeros((2,Nw))
    tw   = np.zeros((2,Nw)) # arrays of x,y points and tangents on wake
    xy1  = M.foil.x[:,0] 
    xyN  = M.foil.x[:,-1] # airfoil TE points
    xyte = 0.5*(xy1 + xyN) # TE midpoint 
    n    = xyN-xy1 
    t    = np.array([n[1],-n[0]]) # normal and tangent
    if t[0] > 0:  
        assert('Wrong wake direction ensure airfoil points are CCW')
    xyw[:,0] = xyte + 1E-5*t*M.geom.chord # first wake point, just behind TE
    sw = S[-1] + sv # s-values on wake, measured as continuation of the airfoil

    # loop over rest of wake
    for i in range(Nw-1):
        v1         = inviscid_velocity(M.foil.x, M.isol.gam, Vinf, M.oper.alpha, xyw[:,i])
        v1         = v1/np.linalg.norm(v1) 
        tw[:,i]    = v1 # normalized
        xyw[:,i+1] = xyw[:,i] + (sv[i+1]-sv[i])*v1 # forward Euler (predictor) step
        v2         = inviscid_velocity(M.foil.x, M.isol.gam, Vinf, M.oper.alpha, xyw[:,i+1])
        v2         = v2/np.linalg.norm(v2) 
        tw[:,i+1]  = v2 # normalized
        xyw[:,i+1] = xyw[:,i] + (sv[i+1]-sv[i])*0.5*(v1+v2) # corrector step
    

    # determine inviscid ue in the wake, and 0,90deg ref ue too
    uewi    = np.zeros((Nw,1)) 
    uewiref = np.zeros((Nw,2))
    for i in range(Nw):
        v            = inviscid_velocity(M.foil.x, M.isol.gam, Vinf, M.oper.alpha, xyw[:,i])
        uewi[i]      = np.dot(v,tw[:,i])
        v            = inviscid_velocity(M.foil.x, M.isol.gamref[:,0], Vinf, 0, xyw[:,i])
        uewiref[i,0] = np.dot(v,tw[:,i])
        v            = inviscid_velocity(M.foil.x, M.isol.gamref[:,1], Vinf, np.pi/2, xyw[:,i])
        uewiref[i,1] = np.dot(v,tw[:,i])

    # set values
    M.wake.N       = Nw
    M.wake.x       = xyw
    M.wake.s       = sw
    M.wake.t       = tw
    M.isol.uewi    = uewi
    M.isol.uewiref = uewiref

    return 


#-------------------------------------------------------------------------------
def stagpoint_find(M):
    # finds the LE stagnation point on the airfoil (using inviscid solution)
    # INPUTS
    #   M  : mfoil class with inviscid solution, gam
    # OUTPUTS
    #   M.isol.sstag   : scalar containing s value of stagnation point
    #   M.isol.sstag_g : linearization of sstag w.r.t gamma (1xN)
    #   M.isol.Istag   : [i,i+1] node indices before/after stagnation (1x2)
    #   M.isol.sgnue   : sign conversion from CW to tangential velocity (1xN)
    #   M.isol.xi      : distance from stagnation point at each node (1xN)

    if np.shape(M.isol.gam) != 0:
        print('No inviscid solution to rebuild')
    N              = M.foil.N  # number of points on the airfoil
    J              = np.where(M.isol.gam>0)[0] 
    if np.shape(J) == 0:
        print('no stagnation point')
    I              = [J[0]-1, J[0]] 
    G              = M.isol.gam[I]
    S              = M.foil.s[I]
    M.isol.Istag   = I  # indices of neighboring gammas
    den            = (G[1]-G[0]) 
    w1             = G[1]/den 
    w2             = -G[0]/den
    sst            = w1*S[0] + w2*S[1]  # s location
    M.isol.sstag   = sst 
    W_vec          = np.array([w1,w2])
    M.isol.xstag   = np.dot(M.foil.x[:,I],W_vec.T)  # x location
    st_g1          = G[1]*(S[0]-S[1])/(den*den)
    M.isol.sstag_g =  np.array([st_g1, -st_g1])
    sgnue          = -1*np.ones(N)
    sgnue[J]       = 1 # upper/lower surface sign
    M.isol.sgnue   = sgnue
    M.isol.xi      = np.concatenate((abs(M.foil.s-M.isol.sstag), M.wake.s-M.isol.sstag),axis = 0)

    return 


#-------------------------------------------------------------------------------
def rebuild_isol(M):
    # rebuilds inviscid solution, after an angle of attack change
    # INPUT
    #   M     : mfoil class with inviscid reference solution and angle of attack
    # OUTPUT
    #   M.isol.gam : correct combination of reference gammas
    #   New stagnation point location if inviscid
    #   New wake and source influence matrix if viscous
    if np.shape(M.isol.gam) != 0:
        print('No inviscid solution to rebuild')
    print('\n  Rebuilding the inviscid solution.\n')
    alpha      = M.oper.alpha
    M.isol.gam = M.isol.gamref[:,0]*np.cos(alpha) + M.isol.gamref[:,1]*np.sin(alpha)
    if not (M.oper.viscous):
        # viscous stagnation point movement is handled separately
        stagpoint_find(M)
    elif (M.oper.redowake):
        build_wake(M)
        identify_surfaces(M)
        calc_ue_m(M) # rebuild matrices due to changed wake geometry
    
    return 

#-------------------------------------------------------------------------------
def panel_linvortex_velocity(Xj, xi, vdir, onmid):
    # calculates the velocity coefficients for a linear vortex panel
    # INPUTS
    #   Xj    : X(:,[1,2]) = panel point coordinates
    #   xi    : control point coordinates (2x1)
    #   vdir  : direction of dot product
    #   onmid : true means xi is on the panel midpoint
    # OUTPUTS
    #   a,b   : velocity influence coefficients of the panel
    # DETAILS
    #   The velocity due to the panel is then a*g1 + b*g2
    #   where g1 and g2 are the vortex strengths at the panel points
    #   If vdir is [], a,b are 2x1 vectors with velocity components
    #   Otherwise, a,b are dotted with vdir

    # panel coordinates
    xj1 = Xj[0,0]  
    zj1 = Xj[1,0]
    xj2 = Xj[0,1]  
    zj2 = Xj[1,1]

    # panel-aligned tangent and np.linalg.normal vectors
    t = np.array([xj2-xj1 , zj2-zj1])
    t = t/np.linalg.norm(t)
    n = np.array([-t[1] ,t[0]]) 

    # control point relative to (xj1,zj1)
    xz = np.array([(xi[0]-xj1), (xi[1]-zj1)]).T
    x = np.dot(xz,t)  # in panel-aligned coord system
    z = np.dot(xz,n)   # in panel-aligned coord system

    # distances and angles
    d = np.linalg.norm([xj2-xj1, zj2-zj1]) # panel length
    r1 = np.linalg.norm([x,z])             # left edge to control point
    r2 = np.linalg.norm([x-d,z])           # right edge to control point
    theta1 = math.atan2(z,x)          # left angle
    theta2 = math.atan2(z,x-d)        # right angle

    # velocity in panel-aligned coord system
    if (onmid):
        ug1 = 1/2 - 1/4   
        ug2 = 1/4
        wg1 = -1/(2*np.pi)   
        wg2 = 1/(2*np.pi)
    else:
        temp1 = (theta2-theta1)/(2*np.pi)
        temp2 = (2*z*np.log(r1/r2) - 2*x*(theta2-theta1))/(4*np.pi*d)
        ug1 =  temp1 + temp2
        ug2 =        - temp2
        temp1 = np.log(r2/r1)/(2*np.pi)
        temp2 = (x*np.log(r1/r2) - d + z*(theta2-theta1))/(2*np.pi*d)
        wg1 =  temp1 + temp2
        wg2 =        - temp2  
    

    # velocity influence in original coord system
    a = np.array([ug1*t[0]+wg1*n[0], ug1*t[1]+wg1*n[1]]) # point 1
    b = np.array([ug2*t[0]+wg2*n[0], ug2*t[1]+wg2*n[1]]) # point 2
    if np.shape(vdir)[0] != 0:
        a = a.T*vdir
        b = b.T*vdir 

    return a,b


#-------------------------------------------------------------------------------
def panel_constsource_velocity(Xj, xi, vdir):
    # calculates the velocity coefficient for a constant source panel
    # INPUTS
    #   Xj    : X(:,[1,2]) = panel point coordinates
    #   xi    : control point coordinates (2x1)
    #   vdir  : direction of dot product
    # OUTPUTS
    #   a     : velocity influence coefficient of the panel
    # DETAILS
    #   The velocity due to the panel is then a*s
    #   where s is the panel source strength
    #   If vdir is [], a,b are 2x1 vectors with velocity components
    #   Otherwise, a,b are dotted with vdir

    # panel coordinates
    xj1 = Xj[0,0]  
    zj1 = Xj[1,0]
    xj2 = Xj[0,1]  
    zj2 = Xj[1,1]

    # panel-aligned tangent and np.linalg.normal vectors
    t = np.array([xj2-xj1 , zj2-zj1])
    t = t/np.linalg.norm(t)
    n = np.array([-t[1] ,t[0]])

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

    ep = 1e-9
    if (r1 < ep):
        logr1 = 0 
        theta1=np.pi 
        theta2=np.pi 
    else:
        logr1 = np.log(r1) 
    if (r2 < ep):
        logr2 = 0 
        theta1=0 
        theta2=0 
    else:
        logr2 = np.log(r2) 


    # velocity in panel-aligned coord system
    u = (0.5/np.pi)*(logr1 - logr2)
    w = (0.5/np.pi)*(theta2-theta1)

    # velocity in original coord system dotted with given vector
    a = np.array([u*t[0]+w*n[0],u*t[1]+w*n[1]])
    if np.shape(vdir)[0]!= 0:
        a = np.dot(a,vdir)
    return a



#-------------------------------------------------------------------------------
def panel_linsource_velocity(Xj, xi, vdir):
    # calculates the velocity coefficients for a linear source panel
    # INPUTS
    #   Xj    : X(:,[1,2]) = panel point coordinates
    #   xi    : control point coordinates (2x1)
    #   vdir  : direction of dot product
    # OUTPUTS
    #   a,b   : velocity influence coefficients of the panel
    # DETAILS
    #   The velocity due to the panel is then a*s1 + b*s2
    #   where s1 and s2 are the source strengths at the panel points
    #   If vdir is [], a,b are 2x1 vectors with velocity components
    #   Otherwise, a,b are dotted with vdir

    # panel coordinates
    xj1 = Xj[0,0]  
    zj1 = Xj[1,0]
    xj2 = Xj[0,1]  
    zj2 = Xj[1,1]

    # panel-aligned tangent and np.linalg.normal vectors
    t = np.array([xj2-xj1 ,zj2-zj1])
    t = t/np.linalg.norm(t)
    n = np.array([-t[1],t[0]])

    # control point relative to (xj1,zj1
    xz = np.array([(xi[0]-xj1), (xi[1]-zj1)])
    x = np.dot(xz,t)  # in panel-aligned coord system
    z = np.dot(xz,n)   # in panel-aligned coord system

    # distances and angles
    d = np.linalg.norm([xj2-xj1, zj2-zj1]) # panel length
    r1 = np.linalg.norm([x,z])             # left edge to control point
    r2 = np.linalg.norm([x-d,z])           # right edge to control point
    theta1 = math.atan2(z,x)          # left angle
    theta2 = math.atan2(z,x-d)        # right angle

    # velocity in panel-aligned coord system
    temp1 = np.log(r1/r2)/(2*np.pi)
    temp2 = (x*np.log(r1/r2) - d + z*(theta2-theta1))/(2*np.pi*d)
    ug1 =  temp1 - temp2
    ug2 =          temp2
    temp1 = (theta2-theta1)/(2*np.pi)
    temp2 = (-z*np.log(r1/r2) + x*(theta2-theta1))/(2*np.pi*d)
    wg1 =  temp1 - temp2
    wg2 =          temp2

    # velocity influence in original coord system
    a = np.array([ug1*t[0]+wg1*n[0], ug1*t[1]+wg1*n[1]]) # point 1
    b = np.array([ug2*t[0]+wg2*n[0], ug2*t[1]+wg2*n[1]]) # point 2
    if np.shape(vdir) !=0: 
        a = np.dot(a,vdir)  
        b = np.dot(b,vdir)          

    return a, b


#-------------------------------------------------------------------------------
def panel_linsource_stream(Xj, xi):
    # calculates the streamdef coefficients for a linear source panel
    # INPUTS
    #   Xj  : X(:,[1,2]) = panel point coordinates
    #   xi  : control point coordinates (2x1)
    # OUTPUTS
    #   a,b : streamdef influence coefficients
    # DETAILS
    #   The streamdef due to the panel is then a*s1 + b*s2
    #   where s1 and s2 are the source strengths at the panel points

    # panel coordinates
    xj1 = Xj[0,0]  
    zj1 = Xj[1,0]
    xj2 = Xj[0,1]  
    zj2 = Xj[1,1]

    # panel-aligned tangent and np.linalg.normal vectors
    t = np.shape([[xj2-xj1],[ zj2-zj1]])
    t = t/np.linalg.norm(t)
    n = np.array([-t[1],t[0]])

    # control point relative to (xj1,zj1)
    xz = np.array([(xi[0]-xj1), (xi[1]-zj1)])
    x = np.dot(xz,t)  # in panel-aligned coord system
    z = np.dot(xz,n)   # in panel-aligned coord system

    # distances and angles
    d = np.linalg.norm([xj2-xj1, zj2-zj1]) # panel length
    r1 = np.linalg.norm([x,z])             # left edge to control point
    r2 = np.linalg.norm([x-d,z])           # right edge to control point
    theta1 = math.atan2(z,x)          # left angle
    theta2 = math.atan2(z,x-d)        # right angle

    # make branch cut at theta = 0
    if (theta1<0): 
        theta1 = theta1 + 2*np.pi 
    if (theta2<0): 
        theta2 = theta2 + 2*np.pi 

    # check for r1, r2 zero
    ep = 1e-9
    if (r1 < ep): 
        logr1 = 0 
        theta1=np.pi 
        theta2=np.pi 
    else: 
        logr1 = np.log(r1) 
    if (r2 < ep): 
        logr2 = 0 
        theta1=0 
        theta2=0 
    else: 
        logr2 = np.log(r2) 

    # streamdef components
    P1 = (0.5/np.pi)*(x*(theta1-theta2) + theta2*d + z*logr1 - z*logr2)
    P2 = x*P1 + (0.5/np.pi)*(0.5*r2**2*theta2 - 0.5*r1**2*theta1 - 0.5*z*d)

    # influence coefficients
    a = P1-P2/d
    b =    P2/d
 
    return a, b 
