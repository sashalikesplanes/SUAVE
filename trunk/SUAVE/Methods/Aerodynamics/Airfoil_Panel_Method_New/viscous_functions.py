
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


 
 
 
# =========================================================
# VISCOUS FUNCTIONS 
# =========================================================
#-------------------------------------------------------------------------------
def calc_ue_m(M):
    # calculates sensitivity matrix of ue w.r.t. transnp.piration BC mass sources
    # INPUT
    #   M : mfoil class with wake already built
    # OUTPUT
    #   M.vsol.sigma_m : d(source)/d(mass) matrix, for computing source strengths
    #   M.vsol.ue_m    : d(ue)/d(mass) matrix, for computing tangential velocity
    # DETAILS
    #   "mass" flow refers to area flow (we exclude density)
    #   sigma_m and ue_m return values at each node (airfoil and wake)
    #   airfoil panel sources are constant strength
    #   wake panel sources are two-np.piece linear
    if np.shape(M.isol.gam) != 0:
        print('No inviscid solution') 
        
    N = M.foil.N 
    Nw = M.wake.N  # number of points on the airfoil/wake
    if Nw>0:
        assert('No wake')

    # Cgam = d(wake uei)/d(gamma)   [Nw x N]   (not sparse)
    Cgam = np.zeros((Nw,N))
    for i in range(Nw):
        v_G   = inviscid_velocity(M.foil.x, M.isol.gam, 0, 0, M.wake.x[:,i],nargout=2)
        Cgam[i,:] = v_G[0,:]*M.wake.t[0,i] + v_G[1,:]*M.wake.t[1,i]
    

    # B = d(airfoil surf streamdef)/d(source)  [(N+1) x (N+Nw-2)]  (not sparse)
    B = np.zeros((N+1,N+Nw-2))  # note, N+Nw-2 = # of panels
    for i in range(N):  # loop over points on the airfoil
        xi = M.foil.x[:,i] # coord of point i
        for j in range(N-1): # loop over airfoil panels
            B[i,j] = panel_constsource_stream(M.foil.x[:,[j,j+1]], xi)
         
        for j in range(Nw-1): # loop over wake panels
            Xj = M.wake.x[:,[j,j+1]] # panel point coordinates
            Xm = 0.5*(Xj[:,0] + Xj[:,1]) # panel midpoint
            Xj =  np.concatenate((np.concatenate((np.atleast_2d(Xj[:,0]).T,np.atleast_2d(Xm).T), axis = 1),np.atleast_2d(Xj[:,1]).T), axis = 1) # left, mid, right coords on panel
            if (j==(Nw-2)):
                Xj[:,2] = 2*Xj[:,2] - Xj[:,1]  # ghost extension at last point
            a,b = panel_linsource_stream(Xj[:,[0,1]], xi) # left half panel
            if (j > 0):
                B[i,N-1+j]   = B[i,N-1+j] + 0.5*a + b
                B[i,N-1+j-1] = B[i,N-1+j-1] + 0.5*a
            else:
                B[i,N-1+j] = B[i,N-1+j] + b
            
            a,b        = panel_linsource_stream(Xj[:,[1,2]], xi) # right half panel
            B[i,N-1+j] = B[i,N-1+j] + a + 0.5*b
            if (j<Nw-2):
                B[i,N-1+j+1] = B[i,N-1+j+1] + 0.5*b
            else:
                B[i,N-1+j] = B[i,N-1+j] + 0.5*b 
    

    # Bp = - inv(AIC) * B   [N x (N+Nw-2)]  (not sparse)
    # Note, Bp is d(airfoil gamma)/d(source)
    Bp = - np.linalg.solve(M.isol.AIC,B)  # this has N+1 rows, but the last one is np.zero
    Bp = Bp[:-1,:]  # trim the last row

    # Csig = d(wake uei)/d(source) [Nw x (N+Nw-2)]  (not sparse)
    Csig = np.zeros((Nw, N+Nw-2))
    for i in range(Nw):
        xi = M.wake.x[:,i] 
        ti = M.wake.t[:,i] # point, tangent on wake

        # first/last airfoil panel effects on i=1 wake point handled separately
        jstart = 0 + (i==0) 
        jend   = N-1 - (i==0)
        for j in range(jstart,jend): # constant sources on airfoil panels
            Csig[i,j] = panel_constsource_velocity(M.foil.x[:,[j,j+1]], xi, ti)
        

        # np.piecewise linear sources across wake panel halves (else singular)
        for j in range(Nw):  # loop over wake points
            I = [max(j-1,0), j, min(j+1,Nw-1)] # left, self, right
            Xj = M.wake.x[:,I] # point coordinates
            Xj[:,0] = 0.5*(Xj[:,0] + Xj[:,1]) # left midpoint
            Xj[:,2] = 0.5*(Xj[:,1] + Xj[:,2]) # right midpoint
            if (j==Nw):
                Xj[:,2] = 2*Xj[:,1] - Xj[:,0]  # ghost extension at last point
            d1 = np.linalg.norm(Xj[:,1]-Xj[:,0]) # left half-panel len
            d2 = np.linalg.norm(Xj[:,2]-Xj[:,1]) # right half-panel len
            if (i==j):
                if (j==0): # first point: special TE system (three panels meet)
                    dl            = np.linalg.norm(M.foil.x[:,1]-M.foil.x[:, 0]) # lower surface panel len
                    du            = np.linalg.norm(M.foil.x[:,-1]-M.foil.x[:,N-2]) # upper surface panel len
                    Csig[i,  0]   = Csig[i,  0] + (0.5/np.pi)*(np.log(dl/d2) + 1) # lower panel effect
                    Csig[i,N-2]   = Csig[i,N-2] + (0.5/np.pi)*(np.log(du/d2) + 1) # upper panel effect
                    Csig[i,N-2+1] = Csig[i,N-2+1] - 0.5/np.pi # self effect
                
                elif(j==Nw-1): # last point: no self effect of last pan (ghost extension)
                    Csig[i,N-1+j-1] = Csig[i,N-1+j-1] + 0 # hence the 0
                
                else: # all other points
                    aa = (0.25/np.pi)*np.log(d1/d2)
                    Csig[i,N-1+j-1] = Csig[i,N-1+j-1] + aa + 0.5/np.pi
                    Csig[i,N-1+j] = Csig[i,N-1+j] + aa - 0.5/np.pi
                
            else:
                if (j==0): # first point only has a half panel on the right
                    a,b           = panel_linsource_velocity(Xj[:,[1,2]], xi, ti)
                    Csig[i,N-2+1] = Csig[i,N-2+1] + b # right half panel effect
                    Csig[i, 0] = Csig[i, 0] + a # lower airfoil panel effect
                    Csig[i,N-2] = Csig[i,N-2] + a # upper airfoil panel effect
                
                elif (j==Nw-1): # last point has a constant source ghost extension
                    a              = panel_constsource_velocity(Xj[:,[0,2]], xi, ti)
                    Csig[i,N+Nw-3] = Csig[i,N+Nw-3] + a # full const source panel effect
                                
                else: # all other points have a half panel on left and right
                    a1,b1         = panel_linsource_velocity(Xj[:,[0,1]], xi, ti) # left half-panel ue contrib
                    a2,b2         = panel_linsource_velocity(Xj[:,[1,2]], xi, ti) # right half-panel ue contrib
                    Csig[i,N-1+j-1] = Csig[i,N-1+j-1] + a1 + 0.5*b1
                    Csig[i,N-1+j]   = Csig[i,N-1+j] + 0.5*a2 + b2
    

    # compute ue_sigma = d(unsigned ue)/d(source) [(N+Nw) x (N+Nw-2)] (not sparse)
    # Df = +/- Bp = d(foil uei)/d(source)  [N x (N+Nw-2)]  (not sparse)
    # Dw = (Cgam*Bp + Csig) = d(wake uei)/d(source)  [Nw x (N+Nw-2)]  (not sparse)
    Dw              = np.matmul(Cgam,Bp) + Csig
    Dw[0,:]         = Bp[-1,:] # ensure first wake point has same ue as TE
    M.vsol.ue_sigma = np.concatenate((Bp,Dw), axis = 0) # store combined matrix

    # build ue_m from ue_sigma, using sgnue
    rebuild_ue_m(M)
    return 


#-------------------------------------------------------------------------------
def rebuild_ue_m(M):
    # rebuilds ue_m matrix after stagnation panel change (new sgnue)
    # INPUT
    #   M : mfoil class with calc_ue_m already called once
    # OUTPUT
    #   M.vsol.sigma_m : d(source)/d(mass) matrix, for computing source strengths
    #   M.vsol.ue_m    : d(ue)/d(mass) matrix, for computing tangential velocity
    # DETAILS
    #   "mass" flow refers to area flow (we exclude density)
    #   sigma_m and ue_m return values at each node (airfoil and wake)
    #   airfoil panel sources are constant strength
    #   wake panel sources are two-np.piece linear
    if np.shape(M.vsol.ue_sigma) == 0:
        assert('Need ue_sigma to build ue_m')

    # Dp = d(source)/d(mass)  [(N+Nw-2) x (N+Nw)]  (sparse)
    N  = M.foil.N 
    Nw = M.wake.N  # number of points on the airfoil/wake
    Dp = np.zeros((N+Nw-2,N+Nw))
    for i in range(N-1):
        ds = M.foil.s[i+1]-M.foil.s[i]
        # Note, at stagnation: ue = K*s, dstar = const, m = K*s*dstar
        # sigma = dm/ds = K*dstar = m/s (separate for each side, +/-)
        Dp[i,[i,i+1]] = M.isol.sgnue[i:i+2]* np.array([-1,1])/ds
    
    for i in range(Nw-1):
        ds = M.wake.s[i+1]-M.wake.s[i]
        Dp[N-1+i,[N+i,N+i+1]] = np.array([-1,1])/ds
    
    M.vsol.sigma_m = Dp

    # sign of ue at all points (wake too)
    sgue = np.concatenate((M.isol.sgnue, np.ones(Nw)), axis = 0)

    # ue_m = ue_sigma * sigma_m [(N+Nw) x (N+Nw)] (not sparse)
    M.vsol.ue_m = np.matmul(np.diag(sgue),np.matmul(M.vsol.ue_sigma,M.vsol.sigma_m))

    return 
#-------------------------------------------------------------------------------
def identify_surfaces(M):
    # identifies lower/upper/wake surfaces
    # INPUT
    #   M  : mfoil class with stagnation point found
    # OUTPUT
    #   M.vsol.Is : cell array of node indices for lower[0], upper[1], wake[2]

    M.vsol.Is[0] = np.arange(M.isol.Istag[0],-1,-1)
    M.vsol.Is[1] = np.arange(M.isol.Istag[1],M.foil.N)
    M.vsol.Is[2] = np.arange((M.foil.N),(M.foil.N+M.wake.N))
    
    return 


#-------------------------------------------------------------------------------
def set_wake_gap(M):
    # sets height (delta*) of dead air in wake
    # INPUT
    #   M  : mfoil class with wake built and stagnation point found
    # OUTPUT
    #   M.vsol.wgap : wake gap at each wake point
    # DETAILS
    #   Uses cubic def to extrapolate the TE gap into the wake
    #   See Drela, IBL for Blunt Trailing Edges, 1989, 89-2166-CP

    _, hTE, dtdx,_,_ = TE_info(M.foil.x)
    flen = 2.5 # len-scale factor
    dtdx = min(max(dtdx,-3/flen), 3/flen) # clip TE thickness slope
    Lw   = flen*hTE
    wgap = np.zeros(M.wake.N)
    for i in range(M.wake.N):
        xib = (M.isol.xi[(M.foil.N)+i] - M.isol.xi[M.foil.N-1])/Lw
        if (xib <= 1):
            wgap[i] = hTE*(1+(2+flen*dtdx)*xib)*(1-xib)**2 
      
    M.vsol.wgap = wgap 
    return 

#-------------------------------------------------------------------------------
def stagpoint_move(M):
    # moves the LE stagnation point on the airfoil using the global solution ue
    # INPUT
    #   M  : mfoil class with a valid solution in M.glob.U
    # OUTPUT
    #   New sstag, sstag_ue, xi in M.isol
    #   Possibly new stagnation panel, Istag, and hence new surfaces and matrices

    N = M.foil.N  # number of points on the airfoil
    I = M.isol.Istag # current adjacent node indices
    ue = M.glob.U[3,:].T# edge velocity
    sstag0 = M.isol.sstag # original stag point location

    newpanel = True # are we moving to a new panel?
    if (ue[I[1]] < 0):
        # move stagnation point up (larger s, new panel)
        print('  Moving stagnation point up\n')
        J =  np.where(ue[I[1]:] > 0)
        I2 = J[0]+I[1]-1
        for j in range(I[1],(I2-1)):
            ue[j] = -ue[j] 
        I = [I2-1, I2] # new panel
    elif (ue[I[0]] < 0):
        # move stagnation point down (smaller s, new panel)
        print('  Moving stagnation point down\n')
        J = np.where(ue[I[0]:-1:1] > 0)[0] 
        I1 = I[0]-J[0]+1
        for j in range((I1+1),I[0]):
            ue[j] = -ue[j] 
        I = [I1, I1+1] # new panel
    else:
        newpanel = False # staying on the current panel
  

    # move point along panel
    ues = ue[I] 
    S   = M.foil.s[I]
    if (ues[0] > 0) and (ues[1] > 0):
        assert('stagpoint_move: velocity error')
    den             = ues[0] + ues[1] 
    w1              = ues[1]/den 
    w2              = ues[0]/den
    M.isol.sstag    = w1*S[0] + w2*S[1]  # s location
    M.isol.xstag    = M.foil.x[:,I]*np.array([[w1],[w2]]).T # x location
    M.isol.sstag_ue = np.array([ues[1], -ues[0]])*(S[1]-S[0])/(den*den)
    print(f'  Moving stagnation point: s={sstag0} -> s={M.isol.sstag}\n')

    # set new xi coordinates for every point
    M.isol.xi = np.concatenate((abs(M.foil.s-M.isol.sstag), M.wake.s-M.isol.sstag),axis =0)

    # matrices need to be recalculated if on a new panel
    if (newpanel):
        print('  New stagnation panel = #d #d\n', I[0], I[1])
        M.isol.Istag  = I # new panel indices
        sgnue         = np.ones((1,N)) 
        sgnue[1:I[0]] = -1
        M.isol.sgnue  = sgnue # new upper/lower surface signs
        identify_surfaces(M) # re-identify surfaces 
        M.glob.U[3,:]  = ue  # sign of ue changed on some points near stag    
        rebuild_ue_m(M)    
    return 


#-------------------------------------------------------------------------------
def solve_viscous(M):
    # solves the viscous system (BL + outer flow concurrently)
    # INPUT
    #   M  : mfoil class with an airfoil
    # OUTPUT
    #   M.glob.U : global solution
    #   M.post   : post-processed quantities

    solve_inviscid(M)    
    M.oper.viscous = True
    init_thermo(M) # thermodynamics
    build_wake(M)    
    stagpoint_find(M) # from the inviscid solution
    identify_surfaces(M)    
    set_wake_gap(M)  # blunt TE dead air extent in wake
    calc_ue_m(M)    
    init_boundary_layer(M) # initialize boundary layer from ue
    stagpoint_move(M) # move stag point, using viscous solution  
    solve_coupled(M) # solve coupled system
    calc_force(M)    
    get_distributions(M)
    return 


#-------------------------------------------------------------------------------
def solve_coupled(M):
    # Solves the coupled inviscid and viscous system
    # INPUT
    #   M  : mfoil class with an inviscid solution
    # OUTPUT
    #   M.glob.U : global coupled solution
    # DETAILS
    #   Inviscid solution should exist, and BL variables should be initialized
    #   The global variables are [th, ds, sa, ue] at every node
    #   th = momentum thickness ds = displacement thickness
    #   sa = amplification factor or sqrt(ctau) ue = edge velocity
    #   Nsys = N + Nw = total number of unknowns
    #   ue is treated as a separate variable for improved solver robustness
    #   The alternative is to eliminate ue, ds and use mass flow (not done here):
    #     Starting point: ue = uinv + D*m -> ue_m = D
    #     Since m = ue*ds, we have ds = m/ue = m/(uinv + D*m)
    #     So, ds_m = diag(1/ue) - diag(ds/ue)*D
    #     The residual linearization is then: R_m = R_ue*ue_m + R_ds*ds_m

    # Newton loop
    nNewton     = M.param.niglob # number of iterations
    M.glob.conv = False
    print('\n <<< Beginning coupled solver iterations >>> \n')
    for iNewton in range(nNewton):

        # set up the global system
        build_glob_sys(M)
        # compute forces
        calc_force(M)
        # convergence check
        norm = np.linalg.norm(M.glob.R, 2)
        print(f'\nNewton iteration {iNewton}, norm = {norm}\n')
        if (norm < M.param.rtol):
            M.glob.conv = True 
            break 

        # solve global system
        solve_glob(M)
        # update the state
        update_state(M)
        # update stagnation point Newton still OK had R_x effects in R_U
        stagpoint_move(M)
        # update transition
        update_transition(M)
    

    if not M.glob.conv:
        print('\n** Global Newton NOT CONVERGED **\n') 




#-------------------------------------------------------------------------------
def update_state(M):
    # updates state, taking into account physical constraints
    # INPUT
    #   M  : mfoil class with a valid solution (U) and proposed update (dU)
    # OUTPUT
    #   M.glob.U : updated solution, possibly with a fraction of dU added
    # DETAILS
    #   U = U + omega * dU omega = under-relaxation factor
    #   Calculates omega to prevent big changes in the state or negative values

    if np.any(np.imag(M.glob.U[2,:])):
        M.glob.U[2,:]
        print('imaginary amp in U') 
    if np.any(np.imag(M.glob.dU[2,:])):
        M.glob.dU[2,:]
        print('imaginary amp in dU') 

    # max ctau
    It    = np.where(M.vsol.turb == 1)[0]
    ctmax = max(M.glob.U[2,It])

    # starting under-relaxation factor
    omega = 1.0

    # first limit theta and delta*
    for k in range(2):  
        Uk  = M.glob.U[k,:] 
        dUk = M.glob.dU[k,:]
        # prevent big decreases in th, ds
        fmin = min(dUk/Uk) # find most negative ratio
        if (fmin < -0.5):
            om = abs(0.5/fmin) 
        else:
            om = 1 
        if (om<omega):
            omega = om 
            print(f'  th/ds decrease: omega = {omega}\n') 
    

    # limit negative amp/ctau
    Uk = M.glob.U[2,:] 
    dUk = M.glob.dU[2,:]
    for i in range(len(Uk)):
        if not (M.vsol.turb[i]) and (Uk[i]<.2):
            continue  # do not limit very small amp (too restrictive)
        if (M.vsol.turb[i]) and (Uk[i]<0.1*ctmax):
            continue  # do not limit small ctau
        if (Uk[i]==0.) or (dUk[i]==0.):
            continue 
        if (Uk[i]+dUk[i] < 0):
            om = 0.8*abs(Uk[i]/dUk[i]) 
            if (om<omega):
                omega = om 
                print(f'  neg sa: omega = {omega}\n')  

    # prevent big changes in amp
    I = np.where(M.vsol.turb == False)[0]  #  find(~M.vsol.turb); 
    if np.any(np.imag(Uk[I])):
        print('imaginary amplification') 
    dumax = max(abs(dUk[I]))
    if (dumax > 0):
        om = abs(2/dumax) 
    else: 
        om = 1 
    if (om<omega):
        omega = om 
        print(f'  amp: omega = {omega}\n') 

    # prevent big changes in ctau
    I = np.where(M.vsol.turb == True)[0]
    dumax = max(abs(dUk[I]))
    if (dumax > 0):
        om = abs(.05/dumax) 
    else: om = 1 
    if (om<omega):
        omega = om 
        print(f'  ctau: omega = {omega}\n') 

    # prevent large ue changes
    dUk = M.glob.dU[3,:]
    fmax = max(abs(dUk)/M.oper.Vinf)
    if (fmax > 0):
        om =.2/fmax 
    else:
        om = 1 
    if (om<omega):
        omega = om 
        print(f'  ue: omega = {omega}\n') 

    # prevent large alpha changes
    if (abs(M.glob.dalpha) > 2):
        omega = min(omega, abs(2/M.glob.dalpha)) 

    # take the update
    print(f'  state update: under-relaxation = {omega}\n') 
    M.glob.U     = M.glob.U + omega*M.glob.dU
    M.oper.alpha = M.oper.alpha + omega*M.glob.dalpha

    # fix bad Hk after the update
    for iss in range(3): # loop over surfaces
        if (iss==2):
            Hkmin = 1.00005 
        else:
            Hkmin = 1.02 
        Is = M.vsol.Is[iss] # surface point indices
        param = build_param(M, iss) # get parameter structure
        for i in range(len(Is)): # loop over points
            j      = Is[i] 
            Uj     = M.glob.U[:,j]
            param  = station_param(M, param, j)
            Hk, _  = get_Hk(Uj, param)
            if (Hk < Hkmin):
                M.glob.U[1,j] = M.glob.U[1,j] + 2*(Hkmin-Hk)*M.glob.U[0,j] 
    

    # fix negative ctau after the update
    for ii in range(len(I)):
        i = It[ii]
        if (M.glob.U[2,i] < 0):
            M.glob.U[2,i] = 0.1*ctmax   
    # rebuild inviscid solution (gam, wake) if angle of attack changed 
    return 


#-------------------------------------------------------------------------------
def jacobian_add_Rx(M):
    # include effects of R_x into R_U: R_ue += R_x*x_st*st_ue
    # INPUT
    #   M  : mfoil class with residual Jacobian calculated
    # OUTPUT
    #   M.glob.R_U : ue linearization updated with R_x
    # DETAILS
    #   The global residual Jacobian has a column for ue sensitivity
    #   ue, the edge velocity, also affects the location of the stagnation point
    #   The location of the stagnation point (st) dictates the x value at each node
    #   The residual also deps on the x value at each node (R_x)
    #   We use the chain rule (formula above) to account for this

    Nsys = M.glob.Nsys # number of dofs
    Iue  = np.arange(4,4*Nsys+1,4)-1 # ue indices in U
    x_st = np.atleast_2d(-M.isol.sgnue).T  # st = stag point [Nsys x 1]
    x_st = np.concatenate((x_st, -np.ones((M.wake.N,1)))) # wake same sens as upper surface
    R_st = np.matmul(M.glob.R_x,x_st)# [3*Nsys x 1]
    Ist  = M.isol.Istag 
    st_ue = M.isol.sstag_ue # stag points, sens
    M.glob.R_U[:,Iue[Ist]] = M.glob.R_U[:,Iue[Ist]] + R_st*st_ue


    return 

#-------------------------------------------------------------------------------
def solve_glob(M):
    # solves global system for the primary variable update dU
    # INPUT
    #   M  : mfoil class with residual and Jacobian calculated
    # OUTPUT
    #   M.glob.dU : proposed solution update
    # DETAILS
    #   Uses the augmented system: fourth residual = ue equation
    #   Supports lift-constrained mode, with an extra equation: cl - cltgt = 0
    #   Extra variable in cl-constrained mode is angle of attack
    #   Solves sparse matrix system for for state/alpha update  

    Nsys = M.glob.Nsys # number of dofs
    docl = M.oper.givencl # 1 if in cl-constrained mode

    # get edge velocity and displacement thickness
    ue    = np.atleast_2d(M.glob.U[3,:]).T
    ds    = np.atleast_2d(M.glob.U[1,:]).T
    uemax = max(abs(ue)) 
    ue    = np.maximum(ue,1e-10*uemax) # avoid 0/negative ue

    # use augmented system: variables = th, ds, sa, ue

    # inviscid edge velocity on the airfoil and wake
    ueinv = get_ueinv(M)
    
    # initialize the global variable Jacobian (TODO: estimate nnz)
    R_V = np.zeros((4*Nsys+docl,4*Nsys+docl)) # +1 for cl-alpha constraint

    # state indices in the global system
    Ids = np.arange(2,4*Nsys,4)-1 # delta star indices
    Iue = np.arange(4,4*Nsys+1,4)-1 # ue indices

    # include effects of R_x into R_U: R_ue += R_x*x_st*st_ue
    jacobian_add_Rx(M)
    
    # assemble the residual
    R = np.vstack((M.glob.R,  ue - (ueinv + np.matmul(M.vsol.ue_m,(ds*ue)))))

    # assemble the Jacobian
    R_V[:3*Nsys,:4*Nsys] = M.glob.R_U
    I                    = np.arange((3*Nsys+1)-1,4*Nsys)
    
    
    
    a = np.array([[1,1,1],[2,2,2],[3,3,3]])
    
    b1 = np.eye(Nsys) - M.vsol.ue_m*np.diag(ds)
    b2 =  -M.vsol.ue_m*np.diag(ue)
    
    xi  = np.array([1,1,2,2])
    yi  = np.array([0,2,0,2])
    
    dim_R_V = len(R_V[0])
    dim_b1 = len(b1[0])
    dim_b2 = len(b2[0])
    
    indexes1 = list(I*dim_R_V + Iue) 
    indexes2 = list(I*dim_R_V + Ids)  
    
    # flatten
    flat_R_V = np.reshape(R_V, (dim_R_V*dim_R_V))
    flat_b1 =list( np.reshape(b1, (dim_b1*dim_b1)))
    flat_b2 =list( np.reshape(b2, (dim_b2*dim_b2)))
    
    np.put(flat_R_V,  indexes1 , flat_b1)
    np.put(flat_R_V,  indexes2 , flat_b2)
    
    # reshape
    R_V = np.reshape(flat_R_V, (dim_R_V,dim_R_V))
     

    if (docl):
        # include cl-alpha residual and Jacobian
        Rcla, Ru_alpha, Rcla_U = clalpha_residual(M) 
        R = np.vstack((R,Rcla))
        R_V[I,4*Nsys+1] = Ru_alpha 
        R_V[4*Nsys+1,:] = Rcla_U 

    # solve system for dU, dalpha
    dV = -np.linalg.solve(R_V,R)

    # store dU, reshaped, in M
    M.glob.dU = np.reshape(dV[:4*Nsys],(4,Nsys))
    if (docl):
        M.glob.dalpha = dV() 
 
    return 

#-------------------------------------------------------------------------------
def clalpha_residual(M):
    # computes cl constraint (or just prescribed alpha) residual and Jacobian
    # INPUT
    #   M  : mfoil class with inviscid solution and post-processed cl_alpha, cl_ue
    # OUTPUT
    #   Rcla     : cl constraint residual = cl - cltgt (scalar)
    #   Ru_alpha : lin of ue residual w.r.t. alpha (Nsys x 1)
    #   Rcla_U   : lin of cl residual w.r.t state (1 x 4*Nsys)
    # DETAILS
    #   Used for cl-constrained mode, with alpha as the extra variable
    #   Should be called with up-to-date cl and cl linearizations

    Nsys  = M.glob.Nsys   # number of dofs
    N     = M.foil.N         # number of points (dofs) on airfoil
    alpha = M.oper.alpha # angle of attack (deg)

    if (M.oper.givencl):  # cl is prescribed, need to trim alpha    
        Rcla            = M.post.cl - M.oper.cltgt # cl constraint residual
        Rcla_U          = [np.zeros(1,4*Nsys), M.post.cl_alpha]
        #Rcla_U(4:4:4*N) = M.post.cl_ue # only airfoil nodes affected 
        # Ru = ue - [uinv + ue_m*(ds*ue)], uinv = uinvref*[cos(alpha)sin(alpha)]
        Ru_alpha        = -get_ueinvref(M)*np.array([[-np.sin(alpha)],[np.cos(alpha)]])*np.pi/180
    else:    # alpha is prescribed, easy
        Rcla     = 0 # no residual
        Ru_alpha = np.zeros(Nsys,1) # not really, but alpha is not changing
        Rcla_U    = np.concatenate((np.zeros(1,4*Nsys), 1))
    

    return Rcla, Ru_alpha, Rcla_U


#-------------------------------------------------------------------------------
def build_glob_sys(M):# builds the primary variable global residual system for the coupled problem
    # INPUT
    #   M  : mfoil class with a valid solution in M.glob.U
    # OUTPUT
    #   M.glob.R   : global residual vector (3*Nsys x 1)
    #   M.glob.R_U : residual Jacobian matrix (3*Nsys x 4*Nsys, sparse)
    #   M.glob.R_x : residual linearization w.r.t. x (3*Nsys x Nsys, sparse)
    # DETAILS
    #   Loops over nodes/stations to assemble residual and Jacobian
    #   Transition dicated by M.vsol.turb, which should be consistent with the state
    #   Accounts for wake initialization and first-point similarity solutions
    #   Also handles stagnation point on node via simple extrapolation

    Nsys       = M.glob.Nsys
    M.glob.R   = np.zeros((3*Nsys,1))
    M.glob.R_U = np.zeros((3*Nsys,4*Nsys))
    M.glob.R_x = np.zeros((3*Nsys,Nsys))

    for iss in range(3):  # loop over surfaces
        Is  = M.vsol.Is[iss] # surface point indices
        xi  = np.take(M.isol.xi,Is) # distance from LE stag point
        N   = len(Is) # number of points on this surface
        U   = M.glob.U[:,Is] # [th, ds, sa, ue] states at all points on this surface
        Aux = np.zeros((1,N)) # auxiliary data at all points: [wgap]

        # get parameter structure
        param = build_param(M, iss)

        # set auxiliary data
        if (iss == 2):
            Aux[0,:] = M.vsol.wgap 

        # special case of tiny first xi -- will set to stagnation state later
        if (iss < 1) and (xi[0] < 1e-8*xi[-1]):
            i0 = 1
        else:
            i0 = 0 # i0 indicates the "first" point station
        

        # first point system 
        if (iss < 2):

            # calculate the stagnation state, a def of U1 and U2
            Ip = [i0,i0+1]
            Ust, Ust_U, Ust_x, xst = stagnation_state(U[:,Ip], xi[Ip]) # stag state
            param.turb   = False 
            param.simi   = True  # similarity station flag  
            R1, R1_Ut, _ = residual_station(param, np.array([xst,xst]) , np.concatenate((Ust, Ust),axis = 1), Aux[:,[i0,i0]])
            param.simi   = False
            R1_Ust       = R1_Ut[:,:4] + R1_Ut[:,4:8]
            R1_U         = np.matmul(R1_Ust,Ust_U)
            R1_x         = np.matmul(R1_Ust,Ust_x)
            J            = np.array([Is[i0], Is[i0+1]])

            if (i0 == 1): 
                # i0=1 point landed right on stagnation: set value to Ust
                # print(param, 2, 'hit stagnation!\n')
                Ig                = 3*Is[0] + np.arange(-2,1)
                Jg                = 4*Is[0] + np.arange(-3,1)
                M.glob.R[Ig]      = U[:3,1] - Ust[:2]
                M.glob.R_U[Ig,Jg] = M.glob.R_U[Ig,Jg] + np.eye(3,4)
                Jg                = [4*J[0] + np.arange(-3,1), 4*J[1] + np.arange(-3,1)]
                M.glob.R_U[Ig,Jg] = M.glob.R_U[Ig,Jg] - Ust_U[:3,:]
                M.glob.R_x[Ig,J]  = -Ust_x[:3,:] 
        else:
            # wake initialization
            R1, R1_U, J = wake_sys(M, param)
            R1 = np.atleast_2d(R1).T
            R1_x = np.empty(shape=[0,1]) # no xi depence of first wake residual
            param.turb = True # force turbulent in wake if still laminar
            param.wake = True
        

        # store first point system in global residual, Jacobian
        Ig = 3*(Is[i0]+1)-1 + np.arange(-2,1) #  MATLAB :  3*Is(i0) + (-2:0) 
        M.glob.R[Ig[0]:Ig[-1]+1] = R1
        for j in range(len(J)):
            Jg = 4*(J[j]+1)-1 + np.arange(-3,1)  #  MATLAB : 4*J(j) + (-3:0)
            M.glob.R_U[Ig[0]:Ig[-1]+1,Jg[0]:Jg[-1]+1] = M.glob.R_U[Ig[0]:Ig[-1]+1,Jg[0]:Jg[-1]+1] + R1_U[:, 4*(j+1) + np.arange(-3,1) - 1]
            if np.shape(R1_x)[0] != 0:
                M.glob.R_x[Ig[0]:Ig[-1]+1,J[j]] = M.glob.R_x[Ig[0]:Ig[-1]+1,J[j]] + R1_x[:,j] 
        

        # march over rest of points
        for i in range((i0+1),N):
            Ip = [i-1,i] # two points involved in the calculation
 
            tran = op.xor(int(M.vsol.turb[Is[i-1]][0]), int(M.vsol.turb[Is[i]][0])) # transition flag

            # residual, Jacobian for point i
            if (tran):
                Ri, Ri_U, Ri_x = residual_transition(M, param, xi[Ip], U[:,Ip], Aux[:,Ip])
                store_transition(M, iss, i)
            else:
                Ri, Ri_U, Ri_x = residual_station(param, xi[Ip], U[:,Ip], Aux[:,Ip])
            

            # store point i contribution in global residual, Jacobian
            Ig                    = 3*(Is[i]+1)-1 + np.arange(-2,1)
            Jg1                   = 4*(Is[i-1]+1) - 1 +np.arange(-3,1)
            Jg2                   = 4*(Is[i]+1)-1+np.arange(-3,1)
            
            M.glob.R[Ig[0]:Ig[-1]+1]                        = M.glob.R[Ig[0]:Ig[-1]+1] + Ri
            M.glob.R_U[Ig[0]:Ig[-1]+1,Jg1[0]:Jg1[-1]+1]     = M.glob.R_U[Ig[0]:Ig[-1]+1,Jg1[0]:Jg1[-1]+1]  + Ri_U[:,:4]
            M.glob.R_U[Ig[0]:Ig[-1]+1,Jg2[0]:Jg2[-1]+1]     = M.glob.R_U[Ig[0]:Ig[-1]+1,Jg2[0]:Jg2[-1]+1]  + Ri_U[:,4:8]
            M.glob.R_x[Ig[0]:Ig[-1]+1,Is[Ip]]               = M.glob.R_x[Ig[0]:Ig[-1]+1,Is[Ip]] + Ri_x

            # following transition, all stations will be turbulent
            if (tran):
                param.turb = True 

    return 
    

    # special stagnation point treatment (not currently used)
    # residual_stagnation(M)



#-------------------------------------------------------------------------------
def  stagnation_state(U, x):
    # extrapolates two states in U, first np.ones in BL, to stagnation
    # INPUT
    #   U  : [U1,U2] = states at first two nodes (4x2)
    #   x  : [x1,x2] = x-locations of first two nodes (2x1)
    # OUTPUT
    #   Ust    : stagnation state (4x1)
    #   Ust_U  : linearization of Ust w.r.t. U1 and U2 (4x8)
    #   Ust_x  : linearization of Ust w.r.t. x1 and x2 (4x2)
    #   xst    : stagnation point location ... close to 0
    # DETAILS
    #   fits a quadratic to the edge velocity: 0 at x=0, then through two states
    #   linearly extrapolates other states in U to x=0, from U1 and U2

    # pull off states
    U1 = U[:,0] 
    U2 = U[:,1] 
    x1 = x[0] 
    x2 = x[1]
    dx = x2-x1 
    dx_x = np.array([-1, 1])
    rx = x2/x1 
    rx_x = np.array([-rx,1])/x1

    # linear extrapolation weights and stagnation state
    w1   =  x2/dx
    w1_x = -w1/dx*dx_x + np.array([ 0,1])/dx
    w2   = -x1/dx 
    w2_x = -w2/dx*dx_x + np.array([-1,0])/dx
    Ust  = U1*w1 + U2*w2

    # quadratic extrapolation of the edge velocity for better slope, ue=K*x
    wk1   = rx/dx
    wk1_x = rx_x/dx - wk1/dx*dx_x
    wk2   = -1/(rx*dx) 
    wk2_x = -wk2*(rx_x/rx + dx_x/dx) 
    K     = wk1*U1[3] + wk2*U2[3]
    K_U   = np.array([0,0,0,wk1, 0,0,0,wk2])
    K_x   = U1[3]*wk1_x + U2[3]*wk2_x 

    # stagnation coord cannot be np.zero, but must be small
    xst    = 1e-6
    Ust[3] = K*xst  # linear dep of ue on x near stagnation
    Ust    = np.atleast_2d(Ust).T  # linear dep of ue on x near stagnation
    Ust_U  = np.vstack((np.hstack((w1*np.eye(3,4), w2*np.eye(3,4))),K_U*xst))
    Ust_x  = np.vstack((np.outer(U1[:3],w1_x) + np.outer(U2[:3],w2_x ),K_x*xst))

    return Ust, Ust_U, Ust_x, xst


#-------------------------------------------------------------------------------
def thwaites_init(K, nu):
    # uses Thwaites correlation to initialize first node in stag point flow
    # INPUT
    #   K  : stagnation point constant
    #   nu : kinematic viscosity
    # OUTPUT
    #   th : momentum thickness
    #   ds : displacement thickness
    # DETAILS
    #   ue = K*x -> K = ue/x = stag point flow constant
    #   th**2 = ue**(-6) * 0.45 * nu * int_0**x ue**5 dx = 0.45*nu/(6*K)
    #   ds = Hstag*th = 2.2*th

    th = np.sqrt(0.45*nu/(6*K)) # momentum thickness
    ds = 2.2*th # displacement thickness

    return th, ds


#-------------------------------------------------------------------------------
def wake_sys(M, param):
# constructs residual system corresponding to wake initialization
# INPUT
#   param  : parameters
# OUTPUT
#   R   : 3x1 residual vector for th, ds, sa
#   R_U : 3x12 residual linearization, as three 3x4 blocks
#   J   : indices of the blocks of U in R_U (lower, upper, wake)

    il = M.vsol.Is[0][-1] 
    Ul = M.glob.U[:,il] # lower surface TE index, state
    iu = M.vsol.Is[1][-1] 
    Uu = M.glob.U[:,iu] # upper surface TE index, state
    iw = M.vsol.Is[2][0] 
    Uw = M.glob.U[:,iw] # first wake index, state
    _, hTE,_,_,_ = TE_info(M.foil.x) # trailing-edge gap is hTE

    # Obtain wake shear stress from upper/lower transition if not turb
    param.turb = True
    param.wake = False # calculating turbulent quantities right before wake
    if (M.vsol.turb[il][0]):
        ctl    = Ul[2] 
        ctl_Ul = np.array([0,0,1,0]) # already turb use state
    else:
        ctl, ctl_Ul = get_cttr(Ul, param)  # transition shear stress, lower
    if (M.vsol.turb[iu][0]):
        ctu    = Uu[2] 
        ctu_Uu = np.array([0,0,1,0]) # already turb use state
    else:
        ctu, ctu_Uu = get_cttr(Uu, param)  # transition shear stress, upper
    thsum  = Ul[0] + Uu[0] # sum of thetas
    ctw    = (ctl*Ul[0] + ctu*Uu[0])/thsum # theta-average
    ctw_Ul = (ctl_Ul*Ul[0] + (ctl - ctw)*np.array([1,0,0,0]))/thsum
    ctw_Uu = (ctu_Uu*Uu[0] + (ctu - ctw)*np.array([1,0,0,0]))/thsum

    # residual note, delta star in wake includes the TE gap, hTE
    R     = np.array([Uw[0]-(Ul[0]+Uu[0]),
                  Uw[1]-(Ul[1]+Uu[1]+hTE),
                  Uw[2]-ctw[0]])
    J    = np.array([il, iu, iw]) # R deps on states at these nodes
    R_Ul = np.vstack((-np.eye(2,4),-ctw_Ul))
    R_Uu = np.vstack((-np.eye(2,4),-ctw_Uu))
    R_Uw = np.eye(3,4)
    R_U  = np.concatenate((np.concatenate((R_Ul, R_Uu),axis =1), R_Uw),axis =1)

    return R, R_U, J


#-------------------------------------------------------------------------------
def wake_init(M, ue):
    # initializes the first point of the wake, using data in M.glob.U
    # INPUT
    #   ue  : edge velocity at the wake point
    # OUTPUT
    #   Uw  : 4x1 state vector at the wake point

    iw     = M.vsol.Is[2][0]
    Uw     = M.glob.U[:,iw] # first wake index, state
    R,_, _ = wake_sys(M, M.param) # construct the wake system
    Uw[:3] = Uw[:3] - R 
    Uw[3]  = ue # solve the wake system, use ue

    return Uw  


#-------------------------------------------------------------------------------
def init_boundary_layer(M):# initializes BL solution on foil and wake by marching with given edge vel, ue
    # INPUT
    #   The edge velocity field ue must be filled in on the airfoil and wake
    # OUTPUT
    #   The state in M.glob.U is filled in for each point

    Hmaxl = 3.8 # above this shape param value, laminar separation occurs
    Hmaxt = 2.5 # above this shape param value, turbulent separation occurs 

    ueinv = get_ueinv(M) # get inviscid velocity
    M.glob.Nsys = M.foil.N + M.wake.N # number of global variables (nodes)

    # do we need to initialize?
    if not (M.oper.initbl) and np.shape( np.empty(shape=[1,0]))[1]==M.glob.Nsys:
        print('\n <<< Starting with current boundary layer >>> \n')
        M.glob.U[3,:] = ueinv # do set a new edge velocity
        return
    

    print('\n <<< Initializing the boundary layer >>> \n')

    M.glob.U = np.zeros((4,M.glob.Nsys)) # global solution matrix
    M.vsol.turb = np.zeros((M.glob.Nsys,1)) # node flag: 0 = laminar, 1 = turbulent

    for iss in range(3):  # loop over surfaces

        #vprint(M.param, 3, '\nSide is = #d:\n', iss)

        Is  = M.vsol.Is[iss] # surface point indices
        xi  = np.take(M.isol.xi,Is) # distance from LE stag point
        ue  = np.take(ueinv,Is) # edge velocities
        N   = len(Is) # number of points
        U   = np.zeros((4,N)) # states at all points: [th, ds, sa, ue]
        Aux = np.zeros((1,N)) # auxiliary data at all points: [wgap]

        # ensure edge velocities are not tiny
        uemax = max(abs(ue)) 
        ue    = np.maximum(ue,1e-8*uemax)

        # get parameter structure
        param = build_param(M, iss)

        # set auxiliary data
        if (iss == 2):
            Aux[0,:] = M.vsol.wgap 

        # initialize state at first point
        i0 = 0
        if (iss < 2): 

            # Solve for the stagnation state (Thwaites initialization + Newton)
            if (xi[0]<1e-8*xi[-1]):
                K       = ue[1]/xi[1] 
                hitstag = True 
            else:
                K       = ue[0]/xi[0] 
                hitstag = False
            
            th, ds  = thwaites_init(K, param.mu0/param.rho0)
            xst     = 1e-6 # small but nonnp.zero
            Ust     = np.array([[th[0]],[ds[0]],[0],[K*xst]])
            nNewton = 20
            for iNewton in range(nNewton): 
                # call residual at stagnation
                param.turb = False 
                param.simi = True  # similarity station flag 
                R, R_U, _  = residual_station(param, np.array([xst,xst]), np.concatenate((Ust,Ust),axis=1), np.zeros((1,2)))
                param.simi = False
                if (np.linalg.norm(R) < 1e-10): 
                    break 
                ID =  np.arange(3)
                A  = R_U[:, ID+4] + R_U[:,ID]
                b  = -R  
                dU = np.concatenate((np.linalg.solve(A,b),np.zeros((1,1))), axis = 0)
                # under-relaxation
                dm = max(abs( np.array([abs(dU[0]/Ust[0])[0],abs(dU[1]/Ust[1])[0]]) ))
                omega = 1 
                if (dm > 0.2):
                    omega = 0.2/dm 
                dU = dU*omega
                Ust = Ust + dU
            

            # store stagnation state in first one (rarely two) points
            if (hitstag):
                U[:,0] = Ust 
                U[3,0] = ue[0] 
                i0=2 
                
            U[:,i0] = Ust[:,0] 
            U[3,i0] = ue[i0]

        else: # wake
            U[:,0]             = wake_init(M, ue[0]) # initialize wake state properly
            param.turb         = True # force turbulent in wake if still laminar
            M.vsol.turb[Is[0]] = True # wake starts turbulent 
        # march over rest of points
        tran = False # flag indicating that we are at transition
        i = i0+1
        while (i<=(N-1)):
            Ip     = [i-1,i] # two points involved in the calculation
            U[:,i] = U[:,i-1] 
            U[3,i] = ue[i] # guess = same state, new ue
            if (tran): # set shear stress at transition interval
                ct, _ = get_cttr(U[:,i], param) 
                U[2,i] = ct
            
            M.vsol.turb[Is[i]] = (tran or param.turb) # flag node i as turbulent
            direct   = True # default is direct mode  
            nNewton  = 30
            iNswitch = 11
            for iNewton in range(nNewton):

                # call residual at this station
                if (tran): # we are at transition
                    #vprint(param, 4, 'i=#d, residual_transition (iNewton = #d) \n', i, iNewton)
                    try:
                        R, R_U, _ = residual_transition(M, param, xi[Ip], U[:,Ip], Aux[:,Ip])
                    except: 
                        print('Transition calculation failed in BL init. Continuing.')
                        M.vsol.xt = 0.5*np.sum(xi[Ip])
                                            
                        U[:,i] = U[:,i-1] 
                        U[3,i] = ue[i] 
                        U[2,i] = ct
                        R = 0 # so we move on
                    
                else:
                    #vprint(param, 4, 'i=#d, residual_station (iNewton = #d)\n', i, iNewton)
                    R, R_U, _  = residual_station(param, xi[Ip], U[:,Ip], Aux[:,Ip])
                
                if (np.linalg.norm(R) < 1e-10):   
                    break 

                if (direct): # direct mode => ue is prescribed => solve for th, ds, sa
                    ID =  np.arange(3)
                    A  = R_U[:, ID+4]
                    b  = -R  
                    dU = np.concatenate((np.linalg.solve(A,b),np.zeros((1,1))), axis = 0)
                else: # inverse mode => Hk is prescribed 
                    Hk, Hk_U = get_Hk(U[:,i], param)
                    A  = np.vstack((R_U[:, 4:8],Hk_U))
                    b  = np.vstack((-R,Hktgt-Hk))
                    dU = np.linalg.solve(A,b)
                

                # under-relaxation
                dm = max(abs(np.array([abs(dU[0]/U[0,i-1])[0], abs(dU[1]/U[1,i-1])[0]])))
                if not (direct):
                    dm = max(dm, abs(dU[3]/U[3,i-1])) 
                if (param.turb):
                    dm = max(dm, abs(dU[2]/U[2,i-1]))
                elif (direct):
                    dm = max(dm, abs(dU[2]/10)) 
                
                omega = 1 
                if (dm > 0.3):
                    omega = 0.3/dm 
                dU = dU*omega

                # trial update
                Ui = U[:,i] + dU[:,0]

                # clip extreme values
                if (param.turb):
                    Ui[2] = max(min(Ui[2], .3), 1e-7) 
                #Hklim = 1.02 if (param.wake), Hklim = 1.00005 
                #[Hk,Hk_U] = get_Hk(Ui, param)
                #dH = max(0,Hklim-Hk) Ui[1] = Ui[1] + dH*Ui[0]

                # check if about to separate
                Hmax = Hmaxl 
                if (param.turb):
                    Hmax = Hmaxt 
                Hk,_ = get_Hk(Ui, param)

                if (direct) and ((Hk>Hmax) or (iNewton > iNswitch)):
                    # no update need to switch to inverse mode: prescribe Hk
                    direct = False
                    #vprint(param, 2, '** switching to inverse: i=#d, iNewton=#d\n', i, iNewton)
                    Hk,_  = get_Hk(U[:,i-1], param) 
                    Hkr = (xi[i]-xi[i-1])/U[0,i-1]
                    if (param.wake):
                        H2 = Hk 
                        for k in range(6):
                            H2 = H2 - (H2+.03*Hkr*(H2-1)**3-Hk)/(1+.09*Hkr*(H2-1)**2) 
                        Hktgt = max(H2, 1.01)
                    elif(param.turb):
                        Hktgt = Hk - .15*Hkr # turb: decrease in Hk
                    else:
                        Hktgt = Hk + .03*Hkr # lam: increase in Hk 
                    
                    if not (param.wake):
                        Hktgt = max(Hktgt, Hmax) 
                    if (iNewton > iNswitch):
                        U[:,i] = U[:,i-1] 
                        U[3,i] = ue[i]  # reinit
                else:
                    U[:,i] = Ui  # take the update
                
            
            if (iNewton >= nNewton-1):
                print( f'** BL init not converged: is={iss}, i={i} **\n') 
                # extrapolate values
                U[:,i] = U[:,i-1] 
                U[3,i] = ue[i]
                if (iss<2):
                    U[0,i] = U[0,i-1]*(xi[i]/xi[i-1])**.5
                    U[1,i] = U[1,i-1]*(xi[i]/xi[i-1])**.5
                else:
                    rlen = (xi[i]-xi[i-1])/(10*U[1,i-1])
                    U[1,i] = (U[1,i-1] + U[0,i-1]*rlen)/(1+rlen)
                
            

            # check for transition
            if not (param.turb) and not (tran) and (U[2,i]>param.ncrit):
                print(f'Identified transition at (is={iss}, i={i}): n={U[2,i]}, ncrit={param.ncrit}\n')
                tran = True 
                continue # redo station with transition 

            if (tran):
                store_transition(M, iss, i)  # store transition location
                param.turb = True 
                tran = False # turbulent after transition 

            i = i+1 # next point 
        # store states 
        M.glob.U[:,Is] = U 
    
    return  

#-------------------------------------------------------------------------------
def store_transition(M, iss, i):
    # stores xi and x transition locations using current M.vsol.xt 
    # INPUT
    #   is,i : side,station number
    # OUTPUT
    #   M.vsol.Xt stores the transition location s and x values

    xt = M.vsol.xt
    i0 = M.vsol.Is[iss][i-1] 
    i1 = M.vsol.Is[iss][i] # pre/post transition nodes
    xi0 = M.isol.xi[i0] 
    xi1 = M.isol.xi[i1] # xi (s) locations at nodes
    if (i0<=M.foil.N-1) and (i1<=M.foil.N-1):
        assert('Can only store transition on airfoil')
    x0 = M.foil.x[0,i0] 
    x1 = M.foil.x[0,i1] # x locations at nodes 
    xi_location = xt # xi location
    x_location = x0 + (xt-xi0)/(xi1-xi0)*(x1-x0) # x location
    slu = ['lower', 'upper']
    print('  transition on ' + slu[iss] + f' side at x= {x_location}\n')
    M.vsol.Xt[iss,0] = xi_location
    M.vsol.Xt[iss,1] = x_location

    return 

#-------------------------------------------------------------------------------
def update_transition(M):
    # updates transition location using current state
    # INPUT
    #   a valid state in M.glob.U
    # OUTPUT
    #   M.vsol.turb : updated with latest lam/turb flags for each node
    #   M.glob.U    : updated with amp factor or shear stress as needed at each node

    for iss in range(2): # loop over lower/upper surfaces

        Is = M.vsol.Is[iss] # surface point indices
        N = len(Is) # number of points

        # get parameter structure
        param = build_param(M, iss)

        # current last laminar station
        I = np.where(np.take(M.vsol.turb,Is) == 0)[0]
        ilam0 = I[-1]

        # current amp/ctau solution (so we do not change it unnecessarily)
        sa = M.glob.U[1,Is]

        # march amplification equation to get new last laminar station
        ilam = march_amplification(M, iss)

        if (ilam == ilam0):
            M.glob.U[1,Is] = sa 
            continue  # no change

        #vprint(param, 2, '  Update transition: last lam [#d]->[#d]\n', ilam0, ilam)

        if (ilam < ilam0):
            # transition is now earlier: fill in turb between [ilam+1, ilam0]
            param.turb = True
            sa0, _ = get_cttr(M.glob.U[:,Is[ilam+1]], param)
            sa1 = sa0 
            if (ilam0<N-1):
                sa1 = M.glob.U[2,Is[ilam0+1]]
            xi = np.take(M.isol.xi,Is)
            dx = xi[min(ilam0+1,N-1)]-xi[ilam+1]
            for i in range((ilam+1),ilam0):
                if (dx==0) or (i==ilam+1):
                    f = 0 
                else:
                    f = (xi[i]-xi[ilam+1])/dx 
                if ((ilam+1) == ilam0):
                    f = 1                     
                M.glob.U[1,Is[i]] = sa0 + f*(sa1-sa0)
                if M.glob.U[1,Is[i]] > 0: 
                    assert('negative ctau in update_transition')
                M.vsol.turb[Is[i]] = 1
                  

        elif (ilam > ilam0):
            # transition is now later: lam already filled in leave turb alone
            for i in range(ilam0,ilam):
                M.vsol.turb[Is[i]] = 0 
        
    return 




#-------------------------------------------------------------------------------
def march_amplification(M, iss):
    # marches amplification equation on surface is
    # INPUT
    #   is : surface number index
    # OUTPUT
    #   ilam : index of last laminar station before transition
    #   M.glob.U : updated with amp factor at each (new) laminar station

    Is    = M.vsol.Is[iss] # surface point indices
    N     = len(Is) # number of points
    param = build_param(M, iss) # get parameter structure
    U     = M.glob.U[:,Is] # states
    turb  = np.take(M.vsol.turb,Is) # turbulent station flag

    # loop over stations, calculate amplification
    U[2,0]     = 0. # no amplification at first station
    param.turb = False 
    param.wake = False
    i          = 1
    while (i <= N-1):
        U1 = U[:,i-1] 
        U2 = U[:,i] # states                  
        if (turb[i]):
            U2[2] = U1[2]*1.01  # initialize amp if turb
        dx = M.isol.xi[Is[i]] - M.isol.xi[Is[i-1]]# interval len

        # Newton iterations, only needed if adding extra amplification in damp
        nNewton = 20
        for iNewton in range(nNewton):
            # amplification rate, averaged
            damp1, damp1_U1 = get_damp(U1, param)
            damp2, damp2_U2 = get_damp(U2, param)
            damp, damp_U    = upwind(0.5, 0, damp1, damp1_U1, damp2, damp2_U2)

            Ramp = U2[2] - U1[2] - damp*dx

            if (iNewton > 12):
                print(f'i= {i}, iNewton= {iNewton}, sa = [{U1[2]}, {U2[2]}], damp = {damp}, Ramp = {Ramp}\n')
            

            if abs(Ramp)<1e-12:  
                break  # converged
            Ramp_U =  np.array([0,0,-1,0,0,0,1,0]) - damp_U*dx
            dU     = -Ramp/Ramp_U[6]
            omega  = 1 
            dmax = 0.5*(1.01-iNewton/nNewton)
            if (abs(dU) > dmax):
                omega = dmax/abs(dU)  
            U2[2] = U2[2] + omega*dU
                   
        
        if (iNewton >= nNewton):
            print('march amp Newton unconverged!\n') 

        # check for transition
        if (U2[2]>param.ncrit):
            print(f'  march_amplification (is,i={iss},{i}): {U2[2]} is above critical.\n')
            break
        else: 
            M.glob.U[2,Is[i]] = U2[2] # store amplification in M.glob.U      
            U[2,i]            = U2[2] # also store in local copy!
            if (np.imag(U[2,i])):
                print('imaginary amp during march') 
        

        i = i+1 # next station
    

    ilam = i-1 # set last laminar station
    return ilam



#-------------------------------------------------------------------------------
def residual_transition(M, param, x, U, Aux):
    # calculates the combined lam + turb residual for a transition station
    # INPUT
    #   param : parameter structure
    #   x     : 2x1 vector, [x1, x2], containing xi values at the points
    #   U     : 4x2 matrix, [U1, U2], containing the states at the points
    #   Aux   : ()x2 matrix, [Aux1, Aux2] of auxiliary data at the points
    # OUTPUT
    #   R     : 3x1 transition residual vector
    #   R_U   : 3x8 residual Jacobian, [R_U1, R_U2]
    #   R_x   : 3x2 residual linearization w.r.t. x, [R_x1, R_x2]
    # DETAILS
    #   The state U1 should be laminar U2 should be turbulent
    #   Calculates and linearizes the transition location in the process
    #   Assumes linear variation of th and ds from U1 to U2

    # states
    U1 = U[:,0] 
    U2 = U[:,1] 
    sa = U[2,:]
    I1 = np.arange(4)
    I2 = np.arange(4,8)
    Z  = np.zeros(4)

    # interval
    x1 = x[0] 
    x2 = x[1] 
    dx = x2-x1

    # determine transition location (xt) using amplification equation
    xt      = x1 + 0.5*dx # guess
    
    ncrit   = param.ncrit # critical amp factor
    nNewton = 20
    #vprint(param, 3, '  Transition interval = [#.5e, #.5e]\n', x1, x2)
    #  U1, U2
    for iNewton in range(nNewton):
        w2    = (xt-x1)/dx 
        w1    = 1-w2 # weights
        Ut    = w1*U1 + w2*U2
        Ut_xt = (U2-U1)/dx # state at xt
        Ut[2] = ncrit
        Ut_xt[2] = 0. # amplification at transition
        damp1, damp1_U1  = get_damp(U1, param)  
        dampt, dampt_Ut  = get_damp(Ut, param)   
        dampt_Ut[2] = 0.
        Rxt = ncrit - sa[0] - 0.5*(xt-x1)*(damp1 + dampt)
        Rxt_xt = -0.5*(damp1+dampt) - 0.5*(xt-x1)*np.matmul(dampt_Ut,Ut_xt)
        dxt = -Rxt[0]/Rxt_xt[0] 
                
        #vprint(param, 4, '   Transition: iNewton,Rxt,xt = #d,#.5e,#.5e\n', iNewton,Rxt,xt)
        dmax = 0.2*dx*(1.1-iNewton/nNewton)
        
        if (abs(dxt)>dmax):
            dxt = dxt*dmax/abs(dxt) 
        if (abs(Rxt) < 1e-10): 
            break 
        if (iNewton<nNewton-1):
            xt = xt + dxt  

    if (iNewton >= nNewton-1):
        #print('Transition location calculation failed.') 
        pass
    M.vsol.xt = xt # save transition location

    # prepare for xt linearizations
    Rxt_U = -0.5*(xt-x1)* np.hstack((damp1_U1 + dampt_Ut*w1, dampt_Ut*w2)) 
    Rxt_U[2] = Rxt_U[2]-1
    Ut_x1p = (U2-U1)*(w2-1)/dx 
    Ut_x2p = (U2-U1)*(-w2)/dx # at fixed xt
    Ut_x1p[2] = 0 
    Ut_x2p[2] = 0 # amp at xt is always ncrit
    Rxt_x1 = 0.5*(damp1+dampt) - 0.5*(xt-x1)*np.matmul(dampt_Ut,Ut_x1p)
    Rxt_x2 =                   - 0.5*(xt-x1)*np.matmul(dampt_Ut,Ut_x2p)

    # sensitivity of xt w.r.t. U,x from Rxt(xt,U,x) = 0 constraint
    xt_U = -Rxt_U/Rxt_xt  
    xt_U1 = np.take(xt_U,I1) 
    xt_U2 = np.take(xt_U,I2)
    xt_x1 = -Rxt_x1/Rxt_xt 
    xt_x2 = -Rxt_x2/Rxt_xt

    # include derivatives w.r.t. xt in Ut_x1 and Ut_x2
    Ut_x1 = Ut_x1p + Ut_xt*xt_x1
    Ut_x2 = Ut_x2p + Ut_xt*xt_x2

    # sensitivity of Ut w.r.t. U1 and U2
    Ut_U1 = w1*np.eye(4) + np.outer((U2-U1),xt_U1)/dx 
    Ut_U2 = w2*np.eye(4) + np.outer((U2-U1),xt_U2)/dx  

    # laminar and turbulent states at transition
    Utl = Ut 
    Utl_U1 = Ut_U1 
    Utl_U2 = Ut_U2 
    Utl_x1 = Ut_x1 
    Utl_x2 = Ut_x2
    Utl[2] = ncrit 
    Utl_U1[2,:] = Z 
    Utl_U2[2,:] = Z 
    Utl_x1[2] = 0 
    Utl_x2[2] = 0
    Utt = Ut
    Utt_U1 = Ut_U1 
    Utt_U2 = Ut_U2 
    Utt_x1 = Ut_x1 
    Utt_x2 = Ut_x2

    # parameter structure
    param = build_param(M, 0)

    # set turbulent shear coefficient, sa, in Utt
    param.turb = True
    cttr, cttr_Ut = get_cttr(Ut, param)
    Utt[2] = cttr 
    Utt_U1[2,:] = np.matmul(cttr_Ut,Ut_U1)
    Utt_U2[2,:] = np.matmul(cttr_Ut,Ut_U2)
    Utt_x1[2] = np.matmul(cttr_Ut,Ut_x1)
    Utt_x2[2] = np.matmul(cttr_Ut,Ut_x2)

    # laminar/turbulent residuals and linearizations
    param.turb = False
    Rl, Rl_U, Rl_x = residual_station(param, np.array([x1,xt]), np.vstack((U1,Utl)).T, Aux)
    Rl_U1          = Rl_U[:,I1]
    Rl_Utl         = Rl_U[:,I2]
    param.turb     = True
    Rt, Rt_U, Rt_x = residual_station(param, np.array([xt,x2]),np.vstack((Utt,U2)).T, Aux)
    Rt_Utt         = Rt_U[:,I1] 
    Rt_U2          = Rt_U[:,I2]

    # combined residual and linearization
    R = Rl + Rt
    if np.any(np.imag(R)):
        print('imaginary transition residual')
    R_U1 = Rl_U1 +  np.matmul(Rl_Utl,Utl_U1) +  np.outer( Rl_x[:,1],xt_U1) + np.matmul(Rt_Utt,Utt_U1) + np.outer( Rt_x[:,0].T,xt_U1)
    R_U2 =  np.matmul(Rl_Utl,Utl_U2) +  np.outer( Rl_x[:,1],xt_U2) + np.matmul(Rt_Utt,Utt_U2) + Rt_U2 + np.outer( Rt_x[:,0].T,xt_U2)
    R_U  = np.hstack((R_U1, R_U2))
    R_x  = np.vstack((Rl_x[:,0] + Rl_x[:,1]*xt_x1 + Rt_x[:,0]*xt_x1 + np.matmul(Rl_Utl,Utl_x1) + np.matmul(Rt_Utt,Utt_x1), 
            Rt_x[:,1] + Rl_x[:,1]*xt_x2 + Rt_x[:,0]*xt_x2 + np.matmul(Rl_Utl,Utl_x2) + np.matmul(Rt_Utt,Utt_x2))).T  
        
    return R, R_U, R_x

#-------------------------------------------------------------------------------
def residual_station(param, x, U, Aux):
    # calculates the viscous residual at one non-transition station
    # INPUT
    #   param : parameter structure
    #   x     : 2x1 vector, [x1, x2], containing xi values at the points
    #   U     : 4x2 matrix, [U1, U2], containing the states at the points
    #   Aux   : ()x2 matrix, [Aux1, Aux2] of auxiliary data at the points
    # OUTPUT
    #   R     : 3x1 residual vector (mom, shape-param, amp/lag)
    #   R_U   : 3x8 residual Jacobian, [R_U1, R_U2]
    #   R_x   : 3x2 residual linearization w.r.t. x, [R_x1, R_x2]
    # DETAILS
    #   The input states are U = [U1, U2], each with th,ds,sa,ue

    # modify ds to take out wake gap (in Aux) for all calculations below
    U[1,:] = U[1,:] - Aux[0,:]

    # states
    U1 = U[:,0] 
    U2 = U[:,1] 
    Um = 0.5*(U1+U2)
    th = U[0,:] 
    ds = U[1,:] 
    sa = U[2,:] 

    # speed needs compressibility correction
    uk1, uk1_u = get_uk(U1[3],param)
    uk2, uk2_u = get_uk(U2[3],param)

    # np.log changes
    thlog   = np.log(th[1]/th[0])
    thlog_U = np.array([-1/th[0],0,0,0, 1/th[1],0,0,0])
    uelog   = np.log(uk2/uk1)
    uelog_U = np.array([0,0,0,-uk1_u/uk1, 0,0,0,uk2_u/uk2])
    xlog    = np.log(x[1]/x[0]) 
    xlog_x  = np.array([-1/x[0], 1/x[1]])
    dx      = x[1]-x[0] 
    dx_x    = np.array([-1, 1])

    # upwinding factor
    upw, upw_U = get_upw(U1, U2, param)

    # shape parameter
    H1, H1_U1 = get_H(U[:,0])
    H2, H2_U2 = get_H(U[:,1])
    H         = 0.5*(H1+H2)
    H_U       = 0.5*np.concatenate((H1_U1, H2_U2),axis = 0)

    # Hstar = KE shape parameter, averaged
    Hs1, Hs1_U1 = get_Hs(U1, param)
    Hs2, Hs2_U2 = get_Hs(U2, param)
    Hs, Hs_U    = upwind(0.5, 0, Hs1, Hs1_U1, Hs2, Hs2_U2)  

    # np.log change in Hstar
    Hslog = np.log(Hs2/Hs1)
    Hslog_U = np.concatenate((-1/Hs1*Hs1_U1, 1/Hs2*Hs2_U2),axis = 0)

    # similarity station is special: U1 = U2, x1 = x2
    if (param.simi):
        thlog   = 0 
        thlog_U = thlog_U*0
        Hslog   = 0 
        Hslog_U = Hslog_U*0
        uelog   = 1 
        uelog_U = uelog_U*0
        xlog    = 1 
        xlog_x  = np.array([0, 0])
        dx      = 0.5*(x[0]+x[1]) 
        dx_x    =  np.array([0.5,0.5])
    

    # Hw = wake shape parameter
    Hw1, Hw1_U1  = get_Hw(U[:,0], Aux[0,0])
    Hw2, Hw2_U2  = get_Hw(U[:,1], Aux[0,1])
    Hw   = 0.5*(Hw1 + Hw2)
    Hw_U = 0.5*np.concatenate((Hw1_U1, Hw2_U2),axis = 0)

    # set up shear lag or amplification factor equation
    if (param.turb):

        # np.log change of root shear stress coeff
        salog   = np.log(sa[1]/sa[0])
        salog_U = np.array([0,0,-1/sa[0],0, 0,0,1/sa[1],0])

        # BL thickness measure, averaged
        de1, de1_U1 = get_de(U1, param)
        de2, de2_U2 = get_de(U2, param)
        de, de_U    = upwind(0.5, 0, de1, de1_U1, de2, de2_U2)

        # np.linalg.normalized slip velocity, averaged
        Us1, Us1_U1 = get_Us(U1, param)
        Us2, Us2_U2 = get_Us(U2, param)
        Us, Us_U    = upwind(0.5, 0, Us1, Us1_U1, Us2, Us2_U2)

        # Hk, upwinded
        Hk1, Hk1_U1 = get_Hk(U1, param)
        Hk2, Hk2_U2 = get_Hk(U2, param)
        Hk, Hk_U    = upwind(upw, upw_U, Hk1, Hk1_U1, Hk2, Hk2_U2)

        # Re_theta, averaged
        Ret1, Ret1_U1 = get_Ret(U1, param)
        Ret2, Ret2_U2 = get_Ret(U2, param)
        Ret, Ret_U    = upwind(0.5, 0, Ret1, Ret1_U1, Ret2, Ret2_U2)

        # skin friction, upwinded
        cf1, cf1_U1 = get_cf(U1, param)
        cf2, cf2_U2 = get_cf(U2, param)
        cf, cf_U    = upwind(upw, upw_U, cf1, cf1_U1, cf2, cf2_U2)

        # displacement thickness, averaged
        dsa = 0.5*(ds[0] + ds[1])
        dsa_U = 0.5*np.array([0,1,0,0, 0,1,0,0])

        # uq = equilibrium 1/ue * due/dx
        uq, uq_U = get_uq(dsa, dsa_U, cf, cf_U, Hk, Hk_U, Ret, Ret_U, param)

        # cteq = root equilibrium wake layer shear coeficient: (ctau eq)**.5
        cteq1, cteq1_U1 = get_cteq(U1, param)
        cteq2, cteq2_U2 = get_cteq(U2, param)
        cteq, cteq_U    = upwind(upw, upw_U, cteq1, cteq1_U1, cteq2, cteq2_U2)

        # root of shear coefficient (a state), upwinded
        saa, saa_U = upwind(upw, upw_U, sa[0], np.array([0,0,1,0]), sa[1],  np.array([0,0,1,0]))

        # lag coefficient
        Klag = param.SlagK
        beta = param.GB
        Clag = Klag/beta*1/(1+Us)
        Clag_U = -Clag/(1+Us)*Us_U

        # extra dissipation in wake
        ald = 1.0
        if (param.wake):
            ald = param.Dlr 

        # shear lag equation
        Rlag = Clag*(cteq-ald*saa)*dx - 2*de*salog + 2*de*(uq*dx-uelog)*param.Cuq
        Rlag_U = Clag_U*(cteq-ald*saa)*dx + Clag*(cteq_U-ald*saa_U)*dx \
            - 2*de_U*salog - 2*de*salog_U \
            + 2*de_U*(uq*dx-uelog)*param.Cuq + 2*de*(uq_U*dx-uelog_U)*param.Cuq
        Rlag_x = Clag*(cteq-ald*saa)*dx_x + 2*de*uq*dx_x

    else:
        # laminar, amplification factor equation

        if (param.simi):
            # similarity station
            Rlag   = sa[0] + sa[1] # no amplification
            Rlag_U = np.array([0,0,1,0, 0,0,1,0])
            Rlag_x = np.array([0,0])
        else:
            # amplification factor equation in Rlag

            # amplification rate, averaged
            damp1, damp1_U1 = get_damp(U1, param)
            damp2, damp2_U2 = get_damp(U2, param)
            damp, damp_U = upwind(0.5, 0, damp1, damp1_U1, damp2, damp2_U2)

            Rlag = sa[1] - sa[0] - damp*dx
            Rlag_U = np.array([0,0,-1,0, 0,0,1,0]) - damp_U*dx
            Rlag_x = -damp*dx_x 

    # squared mach number, symmetrical average
    Ms1, Ms1_U1 = get_Mach2(U1, param)
    Ms2, Ms2_U2 = get_Mach2(U2, param)
    Ms, Ms_U = upwind(0.5, 0, Ms1, Ms1_U1, Ms2, Ms2_U2)

    # skin friction * x/theta, symmetrical average
    cfxt1, cfxt1_U1, cfxt1_x1 = get_cfxt(U1, x[0], param)
    cfxt2, cfxt2_U2, cfxt2_x2 = get_cfxt(U2, x[1], param)
    cfxtm, cfxtm_Um, cfxtm_xm = get_cfxt(Um, 0.5*(x[0]+x[1]), param)
    cfxt   = 0.25*cfxt1 + 0.5*cfxtm + 0.25*cfxt2
    cfxt_U = 0.25*np.concatenate((cfxt1_U1+cfxtm_Um, cfxtm_Um+cfxt2_U2),axis = 0)
    cfxt_x = 0.25*np.concatenate((cfxt1_x1+cfxtm_xm, cfxtm_xm+cfxt2_x2),axis = 0)

    # momentum equation
    Rmom   = thlog + (2+H+Hw-Ms)*uelog - 0.5*xlog*cfxt
    Rmom_U = thlog_U + (H_U+Hw_U-Ms_U)*uelog + (2+H+Hw-Ms)*uelog_U - 0.5*xlog*cfxt_U
    Rmom_x = -0.5*xlog_x*cfxt - 0.5*xlog*cfxt_x  

    # dissipation def times x/theta: cDi = (2*cD/H*)*x/theta, upwinded
    cDixt1, cDixt1_U1, cDixt1_x1 = get_cDixt(U1, x[0], param)
    cDixt2, cDixt2_U2, cDixt2_x2 = get_cDixt(U2, x[1], param)
    cDixt, cDixt_U               = upwind(upw, upw_U, cDixt1, cDixt1_U1, cDixt2, cDixt2_U2)
    cDixt_x                      = np.concatenate(((1-upw)*cDixt1_x1, upw*cDixt2_x2),axis = 0)

    # cf*x/theta, upwinded
    cfxtu, cfxtu_U = upwind(upw, upw_U, cfxt1, cfxt1_U1, cfxt2, cfxt2_U2)
    cfxtu_x        = np.concatenate(((1-upw)*cfxt1_x1, upw*cfxt2_x2),axis = 0)

    # Hss = density shape parameter, averaged
    Hss1, Hss1_U1  = get_Hss(U1, param)
    Hss2, Hss2_U2  = get_Hss(U2, param)
    Hss, Hss_U     = upwind(0.5, 0, Hss1, Hss1_U1, Hss2, Hss2_U2)

    Rshape   = Hslog + (2*Hss/Hs + 1-H-Hw)*uelog + xlog*(0.5*cfxtu - cDixt)
    Rshape_U = Hslog_U + (2*Hss_U/Hs - 2*Hss/Hs**2*Hs_U -H_U - Hw_U)*uelog + \
        (2*Hss/Hs + 1-H-Hw)*uelog_U + xlog*(0.5*cfxtu_U - cDixt_U)
    Rshape_x = xlog_x*(0.5*cfxtu - cDixt) + xlog*(0.5*cfxtu_x - cDixt_x)

    # put everything together
    R   = np.vstack((np.vstack((Rmom,Rshape)),Rlag))
    R_U = np.vstack((np.vstack((Rmom_U,Rshape_U)),Rlag_U))
    R_x = np.vstack((np.vstack((Rmom_x,Rshape_x)),Rlag_x))   
        
    return R, R_U, R_x

#-------------------------------------------------------------------------------
def residual_stagnation(M):
    # replaces the residual and Jacobian in the global system with the 
    # stagnation residual and Jacobian, at the two stagnation points

    param = build_param(M, 1) # parameters
    param.turb = False 
    param.simi = False 
    param.wake = False


    Ist = M.isol.Istag # stagnation point indices
    I   = [Ist[0]-1, Ist[0], Ist[1], Ist[1]+1] # surrounding points too
    x   = M.isol.xi[I] # x-coords, measured away from stag
    U   = M.glob.U[:,I] # 4 states

    # weights for calculating the stagnation state
    dx   = x[2]+x[1] 
    dx_x = [1,1] # derivatives refer to points 2,3
    w2   =  x[2]/dx 
    w2_x = -w2/dx*dx_x + [ 0,1]/dx
    w3   =  x[1]/dx 
    w3_x = -w3/dx*dx_x + [ 1,0]/dx

    # stagnation state
    Ust    = w2*U[:,1] + w3*U[:,2]
    K      = 0.5*(U[3,1]/x[1] + U[3,2]/x[2]) # ue = K*x at stag
    K_U    = np.array([0,0,0,0.5/x[1], 0,0,0,0.5/x[2]])
    K_x    = 0.5*np.array([-U[3,1]/x[1]**2, -U[3,2]/x[2]**2])
    xst    = 1e-6*M.geom.chord # xst needs to be small but nonnp.zero
    Ust[3] = K*xst
    Ust_U  = np.array([[w2*np.eye(3,4), w3*np.eye(3,4) ],[K_U*xst]])
    Ust_x  = np.array([[U[:3,1]*w2_x + U[:3,2]*w3_x ],[K_x*xst]])

    # ue and x np.log quantities at stagnation (both have to be the same constant)
    uelog = 1
    xlog = 1

    # shape parameter
    H, H_Ust = get_H(Ust)

    # squared Mach number
    Ms, Ms_Ust = get_Mach2(Ust, param)

    # skin friction * x/theta
    cfxt, cfxt_Ust, _ = get_cfxt(Ust, xst, param)

    # momentum equation at stagnation
    Rmom     = (2+H-Ms)*uelog - 0.5*xlog*cfxt
    Rmom_Ust = (H_Ust-Ms_Ust)*uelog - 0.5*xlog*cfxt_Ust
    Rmom_U   = Rmom_Ust*Ust_U
    Rmom_x   = Rmom_Ust*Ust_x

    # dissipation def times x/theta: cDi = (2*cD/H*)*x/theta
    cDixt, cDixt_Ust, _ = get_cDixt(Ust, xst, param)

    # Hstar = KE shape parameter, averaged
    Hs, Hs_Ust = get_Hs(Ust, param)

    # Hss = density shape parameter
    Hss, Hss_Ust = get_Hss(Ust, param)

    # shape parameter equation at stagnation
    Rshape     = (2*Hss/Hs+1-H)*uelog + xlog*(0.5*cfxt - cDixt)
    Rshape_Ust = (2*Hss_Ust/Hs - 2*Hss/Hs**2*Hs_Ust - H_Ust)*uelog + xlog*(0.5*cfxt_Ust - cDixt_Ust)
    Rshape_U   = Rshape_Ust*Ust_U
    Rshape_x   = Rshape_Ust*Ust_x

    # amplification equation at stagnation
    Ramp     = Ust[2] # no amplification
    Ramp_Ust = [0,0,1,0]
    Ramp_U   = Ramp_Ust*Ust_U
    Ramp_x   = Ramp_Ust*Ust_x

    # put stagnation residual into the global system -- Ist[0] eqn
    Ig                 = 3*Ist[0] + np.arange(-2,1)
    Jg                 = [4*I[1]+np.arange(-3,1), 4*I[2]+np.arange(-3,1)]
    M.glob.R[Ig]       = np.array([[Rmom],[ Rshape],[  Ramp]])
    M.glob.R_U[Ig,:]   = 0 
    M.glob.R_x[Ig,:]   = 0 # clear out rows
    M.glob.R_U[Ig,Jg]  = np.array([[Rmom_U ],[Rshape_U ],[Ramp_U]])
    M.glob.R_x[Ig,Ist] = np.array([[Rmom_x ],[Rshape_x],[ Ramp_x]])

    # second equation: second-order BL eqns between points 2 and 3

    # CKx = C/K*dx ue = K*x + 0.5*C*x**2
    ue = U[3,:]
    u12 = ue[0]/ue[1]
    u43 = ue[3]/ue[2]
    x21 = x[1]/x[0] 
    x21_x = np.array([-x21,1,0,0]/x[0])
    x34 = x[2]/x[3] 
    x34_x = np.array([0,0,1,-x34]/x[3])
    dx = x[2]+x[1] 
    dx_x = np.array([0,1,1,0]) # stag interval len note x1,x2 are neg
    dx1 = dx/x[0] 
    dx1_x = dx_x/x[0]-dx/x[0]**2*np.array([1,0,0,0])
    dx4 = dx/x[3] 
    dx4_x = dx_x/x[3]-dx/x[3]**2*np.array([0,0,0,1])
    CKx = (1-u12*x21)*dx1 - (1-u43*x34)*dx4
    CKx_ue = np.array([ [-1, u12]/ue[1]*x21*dx1, [-u43,1]/ue[2]*x34*dx4 ])
    CKx_x = (-u12*x21_x)*dx1 + (1-u12*x21)*dx1_x - (-u43*x34_x)*dx4 - (1-u43*x34)*dx4_x

    # momentum equation
    m0            = (2+H-Ms) # at stagnation
    m0_Ust        = (H_Ust-Ms_Ust)
    th0           = Ust[0] 
    th0_Ust       = np.array([1,0,0,0])
    dth           = U[0,2]-U[0,1] 
    dth_U         = np.array([-1,0,0,0, 1,0,0,0])
    H2, H2_U2     = get_H(U[:,1])
    H3, H3_U3     = get_H(U[:,2])
    Ms2, Ms2_U2   = get_Mach2(U[:,1], param)
    Ms3, Ms3_U3   = get_Mach2(U[:,2], param)
    dm            = (2+H3-Ms3)-(2+H2-Ms2) 
    dm_U          = np.concatenate((-H2_U2+Ms2_U2, H3_U3-Ms3_U3),axis = 0)
    F2, F2_U2     = get_cfutstag(U[:,1], param)
    F3, F3_U3     = get_cfutstag(U[:,2], param)
    dF            = F3-F2 
    dF_U          = np.concatenate((-F2_U2, F3_U3),axis = 0)
    Rmom          = (1+2*m0)*dth/th0 + dm + m0*CKx - 0.5*dF/(K*th0**2)
    Rmom_Ust      = 2*m0_Ust*dth/th0 - (1+2*m0)*dth/th0**2*th0_Ust + m0_Ust*CKx + dF/(K*th0**3)*th0_Ust
    Rmom_U        = (1+2*m0)*dth_U/th0 + dm_U - 0.5*dF_U/(K*th0**2) + 0.5*dF/(K**2*th0**2)*K_U + Rmom_Ust*Ust_U
    Rmom_x        = Rmom_Ust*Ust_x + 0.5*dF/(K**2*th0**2)*K_x
    Rmom_CKx      = m0

    # shape equation
    h0                   = 2*Hss/Hs+1-H # at stagnation
    h0_Ust               = 2*Hss_Ust/Hs - 2*Hss/Hs**2*Hs_Ust - H_Ust
    Hs2, Hs2_U2          = get_Hs(U[:,1], param)
    Hs3, Hs3_U3          = get_Hs(U[:,2], param)
    dHs                  = Hs3-Hs2 
    dHs_U                = np.concatenate((Hs2_U2, Hs3_U3),axis = 0)
    Hss2, Hss2_U2        = get_Hss(U[:,1], param)
    Hss3, Hss3_U3        = get_Hss(U[:,2], param)
    h2                   = 2*Hss2/Hs2+1-H2
    h2_U2                = 2*Hss2_U2/Hs2 - 2*Hss2/Hs2**2*Hs2_U2 - H2_U2
    h3                   = 2*Hss3/Hs3+1-H3
    h3_U3                = 2*Hss3_U3/Hs3 - 2*Hss3/Hs3**2*Hs3_U3 - H3_U3
    dh                   = h3-h2 
    dh_U                 = np.concatenate((-h2_U2, h3_U3),axis = 0)
    D2, D2_U2            = get_cdutstag(U[:,1], param)
    D3, D3_U3            = get_cdutstag(U[:,2], param)
    G2                   = D2-0.5*F2 
    G2_U2                = D2_U2-0.5*F2_U2
    G3                   = D3-0.5*F3 
    G3_U3                = D3_U3-0.5*F3_U3
    dG                   = G3-G2 
    dG_U                 = np.concatenate((-G2_U2, G3_U3),axis = 0)
    Rshape               = th0/Hs*dHs + dh*th0 + h0*(CKx*th0 + 2*dth) - dG/(K*th0)
    Rshape_Ust           = (th0_Ust-th0/Hs*Hs_Ust)/Hs*dHs + dh*th0_Ust + h0_Ust*(CKx*th0+2*dth) + h0*(CKx*th0_Ust) + dG/(K*th0**2)*th0_Ust
    Rshape_U             = th0/Hs*dHs_U + dh_U*th0 + h0*2*dth_U- dG_U/(K*th0) + dG/(K**2*th0)*K_U + Rshape_Ust*Ust_U
    Rshape_x             = Rshape_Ust*Ust_x + dG/(K**2*th0)*K_x
    Rshape_CKx           = h0*th0

    # amplification (none, difference this time)
    Ramp     = U(3,3)-U(3,2)
    Ramp_U   = np.array([0,0,-1,0, 0,0,1,0])
    Ramp_x   = np.array([0,0])
    Ramp_CKx = 0

    # put into the global system
    Ig                 = 3*Ist[1] + np.arange(-2,1)
    #Jg                 = np.array([4*I[1]+(-3:0), 4*I[2]+(-3:0)])
    M.glob.R[Ig]       = np.array([[Rmom],[ Rshape],[ Ramp]])
    M.glob.R_U[Ig,:]   = 0 
    M.glob.R_x[Ig,:]   = 0 # clear out rows
    M.glob.R_U[Ig,Jg]  = np.array([[Rmom_U],[ Rshape_U ],[Ramp_U]])
    M.glob.R_x[Ig,Ist] = np.array([[Rmom_x],[ Rshape_x ],[Ramp_x]])

    Jgue = 4*I
    R_CKx = np.array([[Rmom_CKx],[ Rshape_CKx],[ Ramp_CKx]])
    M.glob.R_U[Ig,Jgue] = M.glob.R_U[Ig,Jgue] + R_CKx*CKx_ue
    M.glob.R_x[Ig,I] = M.glob.R_x[Ig,I] + R_CKx*CKx_x

    return 
