
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
# PLOTTING AND POST PROCESSSING 
# =========================================================  

#-------------------------------------------------------------------------------
def plot_panels(M):
    # plots the airfoil and wake panels
    # INPUT
    #   M : mfoil structure
    # OUTPUT
    #   plot of panels

    fig = plt.figure()
    ax  = fig.add_subplot(1,1,1) 
    ax.plot(M.foil.x[0,:], M.foil.x[1,:], 'bo-', linewidth =  2)  
    if np.shape(M.wake.x)[0] > 0: 
        ax.plot(M.wake.x[0,:], M.wake.x[1,:], 'ro-', linewidth = 2)        
        
    return 

#-------------------------------------------------------------------------------
def plot_stream(M, xrange, npoint):
    # plots the streamfunction contours
    # INPUT
    #   M : mfoil structure
    #   xrange = [xmin, xmax, zmin, zmax] window range
    #   npoint = number of 1d plotting points
    # OUTPUT
    #   streamfunction plot
    if np.shape(M.isol.gam) != 0: 
        assert('No inviscid solution')
    if np.xrange.size == 0:
        xrange = [-.05, 1.05, -.1, .1]  
    sx = np.linspace(xrange[0], xrange[1], npoint)
    sz = np.linspace(xrange[2], xrange[3], npoint)

    [Z,X] = np.meshgrid(sz,sx) 
    P     = X

    N     = M.foil.N         # number of airfoil points
    Nw    = M.wake.N         # number of wake points
    Vinf  = M.oper.Vinf      # freestream speed
    alpha = M.oper.alpha     # angle of attack [deg]
    G     = M.isol.gam       # gamma vector

    _,_,_,tcp,tdp = TE_info(M.foil.x) # trailing-edge info

    # calculate source terms if viscous
    if (M.oper.viscous):
        ue   = M.glob.U[3,:]
        ds   = M.glob.U[1,:]
        sigv = M.vsol.sigma_m*(ue*ds) 

    # loop over plotting points
    for ip in range(len(X[0])):
        for jp in range(len(X[1])): 
            xi  = [X[ip,jp], Z[ip,jp]] 
            Psi = 0

            # panel influence
            for j in range(N-1):
                aij, bij = panel_linvortex_stream(M.foil.x[:,[j,j+1]], xi)
                Psi = Psi + aij*G[j] + bij*G(j+1) 

            # TE source influence
            a = panel_constsource_stream(M.foil.x[:,[N,0]], xi)
            Psi = Psi + a*0.5*tcp*(G[-1]-G[0])

            # TE vortex influence
            [a,b] = panel_linvortex_stream(M.foil.x[:,[N,0]], xi)
            Psi = Psi + (a+b)*0.5*(-tdp)*(G[-1]-G[0])

            # viscous source influence
            if (M.oper.viscous):
                for i in range(N):
                    Psi = Psi + sigv[i]*panel_constsource_stream(M.foil.x[:,[i,i+1]], xi) 
                for i in range(Nw-1):
                    Psi = Psi + sigv[N-1+i]*panel_constsource_stream(M.wake.x[:,[i,i+1]], xi) 

            # add freestream
            P[ip,jp] = Psi + Vinf*np.cos(alpha)*Z[ip,jp] - Vinf*np.sin(alpha)*X[ip,jp] 
    
        
    fig = plt.figure()
    ax  = fig.add_subplot(1,1,1)     
    CS  = ax.contourf(X,Z,P,51)  
    plot_panels(M)
    
    return 
 
#-------------------------------------------------------------------------------
def plot_velocity(M, xrange, npoint):
    # makes a quiver plot of velocity vectors in the domain
    # INPUT
    #   xrange : axes range [xmin, xmax, zmin, zmax]
    #   npoint : number of points (1d)
    # OUTPUT
    #   figure with velocity vectors

    N  = M.foil.N 
    Nw = M.wake.N  # number of points on the airfoil/wake
    if np.shape(xrange) == 0:
        xrange = [-.05, 1.05, -.1, .1] 
    sx = np.linspace(xrange[0], xrange[1], npoint)
    sz = np.linspace(xrange[2], xrange[3], npoint)

    [Z,X] = np.meshgrid(sz,sx) 
    U     = X 
    V     = Z
    # calculate source terms if viscous
    if (M.oper.viscous):
        ue = M.glob.U[3,:]
        ds = M.glob.U[1,:]
        sv = M.vsol.sigma_m*(ue*ds) 
    for ip in range(npoint):
        for jp in range(npoint):
            # global coordinate
            xij = [X[ip,jp], Z[ip,jp]]
            # first the inviscid velocity
            v = inviscid_velocity(M.foil.x, M.isol.gam, M.oper.Vinf, M.oper.alpha, xij)
            if (M.oper.viscous):
                # next, the viscous source contribution
                for i in range(N-1):
                    sigma = sv[i]
                    v = v + sigma*panel_constsource_velocity(M.foil.x[:,[i,i+1]], xij, []) 
                for i  in range(Nw-1):
                    sigma = sv[N-1+i] #dm/ds
                    v = v + sigma*panel_constsource_velocity(M.wake.x[:,[i,i+1]], xij, []) 
            U[ip,jp] = v[0] 
            V[ip,jp] = v[1]  

    fig = plt.figure()
    ax  = fig.add_subplot(1,1,1)      
    ax.quiver(X,Z,U,V)  
    plot_panels(M) 
    
    return 


#-------------------------------------------------------------------------------
def plot_cpplus(M):
    # makes a cp plot with outputs printed
    # INPUT
    #   M : mfoil structure
    # OUTPUT
    #   cp plot on current axes

    fig = plt.figure('CP')
    ax  = fig.add_subplot(1,1,1)    
    chord  = M.geom.chord
    if np.shape(M.wake.x)[0] > 0: 
        xz     = np.hstack((M.foil.x,M.wake.x))  
    else: 
        xz     = M.foil.x
    x      = xz[0,:]  
    xrange = np.array([-.1,1.4])*chord
    if (M.oper.viscous > 0):
        ctype = ['red', 'blue', 'black']
        for iss in range(3):
            Is = M.vsol.Is[iss]
            ax.plot(x[Is], M.post.cp[Is] , color = ctype[iss], linestyle = '-' , linewidth =  2)  
            ax.plot(x[Is], M.post.cpi[Is], color = ctype[iss], linestyle = ':', linewidth =  2) 
        
    else:
        ax.plot(x, M.post.cp, 'b-', linewidth = 2)  
    
    if (M.oper.Ma > 0) and (M.param.cps > (min(M.post.cp)-.2)):
        ax.plot([xrange[0], chord], M.param.cps*np.array([1,1]), 'k--', linewidth =  2)
        #ax.plot( 0.8*chord, M.param.cps-0.1, 'sonic $c_p$', 'interpreter', 'latex', 'fontsize', 18) 
    ax.set_ylabel('c_p') 
    ax.set_ylim([1.2,-1])   # inverse axis  
     
    # output text box 
    print(f'Ma = {M.oper.Ma}' + str( M.oper.Ma))
    print(f'AoA (deg.)$ = {M.oper.alpha}')
    print(f'$c_l$ = {M.post.cl}') 
    print(f'$c_m$ = {M.post.cm}')
    print(f'$c_d$ = {M.post.cd}')
    
    return 
#-------------------------------------------------------------------------------
def  plot_airfoil(M):
    # makes an airfoil plot
    # INPUT
    #   M : mfoil structure
    # OUTPUT
    #   airfoil plot on current axes

    fig = plt.figure()
    ax  = fig.add_subplot(1,1,1)  
    xz  = np.concatenate((M.foil.x, M.wake.x),axis = 1)  
    ax.plot(xz[0,:], xz[1,:], 'k-', linewidth =  1)  

#-------------------------------------------------------------------------------
def  plot_boundary_layer(M):
    # makes a plot of the boundary layer
    # INPUT
    #   M : mfoil structure
    # OUTPUT
    #   boundary layer plot on current axes

    if (M.oper.viscous <= 0):
        return 
    fig  = plt.figure()
    ax   = fig.add_subplot(1,1,1)  
    xz   = np.concatenate((np.atleast_2d(M.foil.x),np.atleast_2d(M.wake.x)) ,axis = 0)
    x    = xz[0,:] 
    N    = M.foil.N
    ds   = M.post.ds # displacement thickness
    rl   = 0.5*(1+(ds[0]-ds[-1])/ds(N+1))
    ru   = 1-rl
    t    = np.concatenate((np.atleast_2d(M.foil.t),np.atleast_2d(M.wake.t)) ,axis = 0)
    n    = np.concatenate((np.atleast_2d(-t[1,:]),np.atleast_2d(t[0,:])) ,axis = 0)
    n    = n/vecnorm(n) # outward normals
    xzd  = xz + n*ds # airfoil + delta*
    ctype = ['red', 'blue', 'black']
    for i in range(3):
        iss = i
        if (iss==3):
            xzd = xz + n*ds*ru 
        if (iss==4):
            xzd = xz - n*ds*rl 
            iss = 3 
        Is = M.vsol.Is[iss]
        ax.plot(xzd[0,Is], xzd[1,Is], color = ctype[iss], linestyle = '-', linewidth = 2)
    return 


#-------------------------------------------------------------------------------
def plot_results(M):
    # makes a summary results plot with cp, airfoil, BL delta, outputs
    # INPUT
    #   M : mfoil structure
    # OUTPUT
    #   summary results plot as a new figure  
  
    # cp plot 
    plot_cpplus(M)

    # airfoil plot 
    plot_airfoil(M)

    # BL thickness
    plot_boundary_layer(M) 


#-------------------------------------------------------------------------------
def calc_force(M):
    # calculates force and moment coefficients
    # INPUT
    #   M : mfoil structure with solution (inviscid or viscous)
    # OUTPUT
    #   M.post values are filled in
    # DETAILS
    #   lift/moment are computed from a panel pressure integration
    #   the cp distribution is stored as well

    chord = M.geom.chord 
    xref  = M.geom.xref # chord and ref moment point 
    Vinf  = M.param.Vinf 
    rho   = M.oper.rho 
    alpha = M.oper.alpha
    qinf  = 0.5*rho*Vinf**2 # dynamic pressure
    N     = M.foil.N # number of points on the airfoil

    # calculate the pressure coefficient at each node
    if (M.oper.viscous):
        ue = np.atleast_2d(M.glob.U[3,:]).T
    else:
        ue = get_ueinv(M) 
    cp, cp_ue  = get_cp(ue, M.param) 
    M.post.cp  = cp
    M.post.cpi,_ = get_cp(get_ueinv(M),M.param) # inviscid cp # lift, moment, near-field pressure cd coefficients by cp integration  
    cl        = 0 
    cl_ue    = np.zeros((N,1))
    cl_alpha = 0
    cm       = 0
    cdpi     = 0  
    for i0 in range(1,N+1):
        i  = i0
        ip = i-1 
        if (i0==N):
            i  = 0 
            ip = N-1 
        x1       = M.foil.x[:,ip] 
        x2       = M.foil.x[:,i] # panel points
        dxv      = x2-x1 
        dx1      = x1-xref
        dx2      = x2-xref
        dx1nds   = dxv[0]*dx1[0]+dxv[1]*dx1[1] # (x1-xref) cross n*ds
        dx2nds   = dxv[0]*dx2[0]+dxv[1]*dx2[1] # (x2-xref) cross n*ds
        dx       = -dxv[0]*np.cos(alpha) - dxv[1]*np.sin(alpha) # minus from CW node ordering
        dz       =  dxv[1]*np.cos(alpha) - dxv[0]*np.sin(alpha) # for drag
        cp1      = cp[ip] 
        cp2      = cp[i]
        cpbar    = 0.5*(cp[ip]+cp[i]) # average cp on the panel
        cl       = cl + dx*cpbar
        I        = [ip,i]  
        cl_ue[I,0] = cl_ue[I,0] + dx*0.5*cp_ue[I,0]
        cl_alpha = cl_alpha + cpbar*(np.sin(alpha)*dxv[0] - np.cos(alpha)*dxv[1])*np.pi/180
        cm       = cm + cp1*dx1nds/3 + cp1*dx2nds/6 + cp2*dx1nds/6 + cp2*dx2nds/3
        cdpi     = cdpi + dz*cpbar
      
    cl              = cl/chord 
    cm              = cm/chord**2 
    cdpi            = cdpi/chord
    M.post.cl       = cl 
    M.post.cl_ue    = cl_ue 
    M.post.cl_alpha = cl_alpha
    M.post.cm       = cm
    M.post.cdpi     = cdpi
  
    # viscous contributions
    cd  = np.array([0])
    cdf = np.array([0])
    if (M.oper.viscous):
  
        # Squire-Young relation for total drag (exrapolates theta from  of wake)
        iw = M.vsol.Is[2][-1] # station at the  of the wake
        U  = M.glob.U[:,iw] 
        H  = U[1]/U[0] 
        ue, _ = get_uk(U[3], M.param) # state
        cd = np.array([2.0])*U[0]*(ue/Vinf)**((5+H)/2.)
    
        # skin friction drag
        Df = 0.
        for iss in range(2):
            Is    = M.vsol.Is[iss] # surface point indices
            param = build_param(M, iss) # get parameter structure
            param = station_param(M, param, Is[0])
            cf1   = 0 #get_cf(M.glob.U(:,Is[0]), param) # first cf value
            ue1   = 0 #get_uk(M.glob.U(4,Is[0]), param)
            rho1  = rho
            x1 = M.isol.xstag
            for i in range(len(Is)): # loop over points
                param = station_param(M, param, Is[i])
                cf2,_   = get_cf(M.glob.U[:,Is[i]], param) # get cf value
                ue2,_   = get_uk(M.glob.U[3,Is[i]], param)
                rho2,_  = get_rho(M.glob.U[:,Is[i]], param)
                x2    = M.foil.x[:,Is[i]]
                dxv   = x2 - x1
                dx    = dxv[0]*np.cos(alpha) + dxv[1]*np.sin(alpha)
                Df    = Df + 0.25*(rho1*cf1*ue1**2 + rho2*cf2*ue2**2)*dx
                cf1   = cf2 
                ue1   = ue2 
                x1    = x2
                rho1  = rho2
          
        
        cdf = Df/(qinf*chord)    
      
    # store results
    M.post.cd = cd 
    M.post.cdf = cdf
    M.post.cdp = cd-cdf
     
    print(f'alpha = {round(M.oper.alpha[0],5)} deg, cl = {round(M.post.cl[0],5)}f, cm = {round(M.post.cm[0],5)}')
    print(f'cdpi = {round(M.post.cdpi[0],5)}, cd = {round(M.post.cd[0],5)}, cdf = {round(M.post.cdf[0],5)}, cdp = { round(M.post.cdp[0],5) }' ) 
    
    return  

 
def get_distributions(M):
    # computes various distributions (quantities at nodes) and stores them in M.post
    # INPUT
    #   M  : mfoil class with a valid solution in M.glob.U
    # OUTPUT
    #   M.post : distribution quantities calculated
    # DETAILS
    #   Relevant for viscous solutions
    
    if np.shape(M.glob.U) != 0: 
        assert('no global solution')

    # quantities already in the global state
    M.post.th  = M.glob.U[0,:] # theta
    M.post.ds  = M.glob.U[1,:] # delta*
    M.post.sa  = M.glob.U[2,:] # amp or ctau
    M.post.ue  = get_uk(M.glob.U[3,:], M.param) # compressible edge velocity 
    M.post.uei = get_ueinv(M) # compressible inviscid edge velocity

    # derived viscous quantities
    N   = M.glob.Nsys 
    cf  = np.zeros((N,1)) 
    Ret = np.zeros((N,1))
    Hk  = np.zeros((N,1))
    for iss in range(2):   # loop over surfaces
        Is    = M.vsol.Is[iss] # surface point indices
        param = build_param(M, iss) # get parameter structure
        for i in range(len(Is)):  # loop over points
            j      = Is[i]
            Uj     = M.glob.U[:,j]
            param  = station_param(M, param, j)
            uk ,_     = get_uk(Uj[3], param) # corrected edge speed
            cfloc, _  = get_cf(Uj, param) # local skin friction coefficient
            cf[j]  = cfloc * uk**2/param.Vinf**2 # free-stream-based cf
            Ret[j],_ = get_Ret(Uj, param) # Re_theta
            Hk[j],_  = get_Hk(Uj, param) # kinematic shape factor 
    
    M.post.cf  = cf 
    M.post.Ret = Ret 
    M.post.Hk  = Hk 
    
    return 

def plot_quantity(M, q, qname):
    # plots a quantity q over lower/upper/wake surfaces
    # INPUT
    #   M     : mfoil class with valid airfoil/wake points
    #   q     : vector of values to plot, on all points (wake too if present)
    #   qname : name of quantity, for axis labeling
    # OUTPUT
    #   figure showing q versus x


    fig = plt.figure()
    ax  = fig.add_subplot(1,1,1)       
    xy    = np.concatenate((np.atleast_2d(M.foil.x),np.atleast_2d(M.wake.x)) ,axis = 0)
    ctype = ['red', 'blue', 'black']
    if np.shape(M.vsol.Is) == 0: 
        ax.plot(xy[0,:], q, ctype[1], marker = 'o', linewidth = 2)
    else:
        sleg = ['lower', 'upper', 'wake']
        for iss in range(3):
            Is = M.vsol.Is[iss]
            ax.plot(xy[0,Is], q[Is], color = ctype[iss] , marker = 'o', linewidth = 2,  label = sleg[iss]) 
         
    return 


def plot_mass(M):
    # plots viscous/mass/source quantities
    # INPUT
    #   M     : mfoil class with valid solution
    # OUTPUT
    #   Plots of mass flow at each node, and source on each panel

    fig   = plt.figure()
    ax    = fig.add_subplot(1,1,1)  
    
    N     = M.foil.N 
    Nw    = M.wake.N  # number of points on the airfoil/wake
    xpan  = np.concatenate((0.5*(M.foil.x[1,1:N-1]+M.foil.x[1,2:N]), 0.5*(M.wake.x[1,1:Nw-1]+M.wake.x[1,2:Nw])),axis = 0)
    ue    = M.glob.U[3,:].T 
    ds    = M.glob.U[1,:].T 
    m     = ue*ds # mass flow at nodes
    sigma = M.vsol.sigma_m*m # source on panels

    plot_quantity(M, m, 'mass flow') 
    sleg   = ['lower', 'upper', 'wake']
    ctype  = ['b-', 'r-', 'k-']
    iss    = 1
    Is     = np.arange((N-1))
    ax.plot(xpan[Is], sigma[Is], color = ctype[iss], marker = 'o', linewidth = 2,  label = sleg[iss]) 
    
    iss  = 3 
    Is   = np.arange(N,(N+Nw-2))
    ax.plot(xpan[Is], sigma[Is], color = ctype[iss], marker = 'o', linewidth = 2,  label = sleg[iss]) 
    #ax.set_xlabel('\xi = distance from stagnation point', 'fontsize', 18)
    #ax.set_ylabel('source', 'fontsize', 18)



def plot_distributions(M):
    # plots viscous distributions
    # INPUT
    #   M  : mfoil class with solution
    # OUTPUT
    #   figures of viscous distributions
 
    get_distributions(M)
    plot_quantity(M, M.post.ue, 'u_e = edge velocity')
    plot_quantity(M, M.post.uei, 'inviscid edge velocity')
    plot_quantity(M, M.post.sa, 'amplification or c_{\tau}**{1/2}')
    plot_quantity(M, M.post.Hk, 'H_k = kinematic shape parameter')
    plot_quantity(M, M.post.ds, '\delta*** = displacement thickness')
    plot_quantity(M, M.post.th, '\theta = momentum thickness')
    plot_quantity(M, M.post.cf, 'c_f = skin friction coefficient')
    plot_quantity(M, M.post.Ret, 'Re_{\theta} = theta Reynolds number')
    plot_mass(M)
    
    return  




