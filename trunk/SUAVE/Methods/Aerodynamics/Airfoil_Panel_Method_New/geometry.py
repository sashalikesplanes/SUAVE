
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
 
from paneling_functions import * 

# =========================================================
# geometry  
# =========================================================  
 
#-------------------------------------------------------------------------------
def geom_flap(M, xzhinge, eta):
    # deploys a flap at hinge location xzhinge, with flap angle eta 
    # INPUTS
    #   M       : mfoil class containing an airfoil
    #   xzhinge : flap hinge location (x,z)
    #   eta     : flap angle, positive = down, degrees
    # OUTPUTS
    #   M.foil.x : modified airfoil coordinates
 
    X  = M.geom.xpoint 
    N  = len(X[1]) # airfoil points
    xh = xzhinge[0]  # x hinge location

    # identify points on flap
    If = np.where(X[0,:]>xh)[0]

    # rotate flap points
    R       = np.array([[np.cos[eta], np.sin[eta]],[-np.sin[eta], np.cos[eta]]])
    X[:,If] = xzhinge + R*(X[:,If]-xzhinge)

    # remove flap points to left of hinge
    I = If(X[0,If]<xh) 
    A = np.arange(N)
    I = A.difference(I)

    # re-assemble the airfoil note, chord length is *not* redefined
    M.geom.xpoint = X[:,I]
    M.geom.npoint = len(M.geom.xpoint[1])

    # repanel
    if (M.foil.N > 0):
        make_panels(M, M.foil.N-1) 
 
    return 


#-------------------------------------------------------------------------------
def geom_addcamber(M, xzcamb):
    # adds camber to airfoil from given coordinates
    # INPUTS
    #   M       : mfoil class containing an airfoil
    #   xzcamb  : (x,z) points on camberline increment, 2 x Nc
    # OUTPUTS
    #   M.foil.x : modified airfoil coordinates

    if len(xzcamb[0]) > len(xzcamb[1]):
        xzcamb = xzcamb.T
    X = M.geom.xpoint # airfoil points

    # interpolate camber delta, add to X 
    f      = np.interp1d(xzcamb[0,:], xzcamb[1,:], kind='cubic') 
    dz     = f(X[0,:])
    X[1,:] = X[1,:] + dz

    # store back in M.geom
    M.geom.xpoint = X 
    M.geom.npoint = len(M.geom.xpoint[1]) 

    # repanel
    if (M.foil.N > 0):
        make_panels(M, M.foil.N-1)  
        
    return 


#-------------------------------------------------------------------------------
def geom_derotate(M):
    # derotates airfoil about leading edge to make chordline horizontal
    # INPUTS
    #   M       : mfoil class containing an airfoil
    # OUTPUTS
    #   M.foil.x : modified airfoil coordinates

    X = M.geom.xpoint
    N = len(X[1]) # airfoil points

    I   = min(X[0,:])
    xLE = X[:,I[0]] # LE point
    xTE = 0.5*(X[:,0] + X[:,N]) # TE "point"

    theta = math.atan2(xTE[1]-xLE[1], xTE[0]-xLE[0]) # rotation angle
    R = [np.cos(theta), np.sin(theta) -np.sin(theta), np.cos(theta)]
    X = xLE + R*(X-xLE) # rotation

    # store back in M.geom
    M.geom.xpoint = X 
    M.geom.npoint = len(M.geom.xpoint[1]) 

    # repanel
    if (M.foil.N > 0):
        make_panels(M, M.foil.N-1)   

    return 

#-------------------------------------------------------------------------------
def space_geom(dx0, L, Np):
    # spaces Np points geometrically from [0,L], with dx0 as first interval
    # INPUTS
    #   dx0 : first interval length
    #   L   : total domain length
    #   Np  : number of points, including points at 0,L
    # OUTPUTS
    #   x   : point locations (1xN)
    if Np>1: 
        assert('Need at least two points for spacing.')
    N = Np - 1 # number of intervals
    # L = dx0 * (1 + r + r**2 + ... r**{N-1}) = dx0*(r**N-1)/(r-1)
    # let d = L/dx0, and for a guess, consider r = 1 + s
    # The equation to solve becomes d*s  = (1+s)**N - 1
    # Initial guess: (1+s)**N ~ 1 + N*s + N*(N-1)*s**2/2 + N*(N-1)*(N-2)*s**3/3
    d = L/dx0 
    a = N*(N-1.)*(N-2.)/6. 
    b = N*(N-1.)/2. 
    c = N-d
    disc = max(b*b-4*a*c, 0.) 
    r = 1 + (-b+np.sqrt(disc))/(2*a)
    for k in range(10):
        R = r**N -1-d*(r-1) 
        R_r = N*r**(N-1)-d 
        dr = -R/R_r
        if (abs(dr)<1e-6): 
            break  
        r = r - R/R_r
    
    vec   = np.arange(N)
    xx    = dx0*r**vec
    x     = np.zeros(N+1)
    x[1:] = np.cumsum(xx)

    return x

#-------------------------------------------------------------------------------
def set_coords(M, X):
    # sets geometry from coordinate matrix
    # INPUTS
    #   M : mfoil class
    #   X : matrix whose rows or columns are (x,z) points, CW or CCW
    # OUTPUTS
    #   M.geom.npoint : number of points
    #   M.geom.xpoint : point coordinates (2 x npoint)
    #   M.geom.chord  : chord length
    # DETAILS
    #   Coordinates should start and  at the trailing edge
    #   Trailing-edge point must be repeated if sharp
    #   Points can be clockwise or counter-clockwise (will detect and make CW)

    if len(X[0]) > len(X[1]):
        X = X.T

    # ensure CCW
    A = 0.
    for i in range(len(X[1])-1):
        A = A + (X(1,i+1)-X(1,i))*(X(2,i+1)+X(2,i)) 
    if (A<0):
        X = np.flip(X) 

    # store points in M
    M.geom.npoint = len(X[1])
    M.geom.xpoint = X  
    M.geom.chord = max(X[0,:]) - min(X[0,:])

    return 


#-------------------------------------------------------------------------------
def naca_points(M, digits):
    # calculates coordinates of a NACA 4-digit airfoil, stores in M.geom
    # INPUTS
    #   M      : mfoil class
    #   digits : character array containing NACA digits
    # OUTPUTS
    #   M.geom.npoint : number of points
    #   M.geom.xpoint : point coordinates (2 x npoint)
    #   M.geom.chord  : chord length
    # DETAILS
    #   Uses analytical camber/thickness formulas
 
    
    # replace with suave functions 
    #M.geom.npoint = len(xs)
    #M.geom.xpoint = [xs zs]  
    #M.geom.chord  #= max(xs) - min(xs)


    return 

#-------------------------------------------------------------------------------
def spline_curvature(Xin, N, Ufac, TEfac):
    # Splines 2D points in Xin and samples using curvature-based spacing 
    # INPUT
    #   Xin   : points to spline
    #   N     : number of points = one more than the number of panels
    #   Ufac  : uniformity factor (1 = normal > 1 means more uniform distribution)
    #   TEfac : trailing-edge resolution factor (1 = normal > 1 = high < 1 = low)
    # OUTPUT
    #   X  : new points (2xN)
    #   S  : spline s values (N)
    #   XS : spline tangents (2xN)

    # min/max of given points (x-coordinate)
    xmin = min(Xin[0,:]) 
    xmax = max(Xin[0,:])

    # spline given points
    PPX, PPY = spline2d(Xin)

    # curvature-based spacing on geom
    nfine = 501
    s = np.linspace(0,PPX.x[-1],nfine)
    xyfine = splineval(PPX, PPY, s)
    PPXfine, PPYfine = spline2d(xyfine)
    s      = PPXfine.x

    sk     = np.zeros(nfine)
    xq, wq = quadseg()
    for i in range(nfine-1):
        ds = s[i+1]-s[i]
        st = xq*ds 
        xss = 6.0*PPXfine.c[0,i]*st + 2.0*PPXfine.c[1,i] 
        yss = 6.0*PPYfine.c[0,i]*st + 2.0*PPYfine.c[1,i]
        skint = 0.01*Ufac+0.5*np.dot(wq, np.sqrt(xss*xss + yss*yss))*ds

        # force TE resolution
        xx = (0.5*(xyfine[0,i]+xyfine[0,i+1])-xmin)/(xmax-xmin) # close to 1 means at TE
        skint = skint + TEfac*0.5*np.exp(-100*(1.0-xx))

        # increment sk
        sk[i+1] = sk[i] + skint
    

    # offset by fraction of average to avoid problems with zero curvature
    sk = sk + 2.0*sum(sk)/nfine

    # arclength values at points
    skl = np.linspace(min(sk), max(sk), N)
    f   = interp1d(sk, s, kind='cubic') 
    s   = f(skl)  

    # new points
    X  = splineval(PPXfine, PPYfine, s) 
    S = s 
    XS = splinetan(PPXfine, PPYfine, s)

    return X, S, XS


#-------------------------------------------------------------------------------
def spline2d(X):
    # splines 2d points
    # INPUT
    #   X : points to spline (2xN)
    # OUTPUT
    #   PP : two-dimensional spline structure 

    N    = len(X[1]) 
    S    = np.zeros(N) 
    Snew = np.zeros(N)

    # estimate the arclength and spline x, y separately
    for i in range(1,N):
        S[i] = S[i-1] + np.linalg.norm(X[:,i]-X[:,i-1])
    PPX = CubicSpline(S,X[0,:])    
    PPY = CubicSpline(S,X[1,:]) 

    # re-integrate to true arclength via several passes
    xq, wq = quadseg()
    for ipass in range(10):
        serr    = 0
        Snew[0] = S[0]
        for i in range(N-1):
            ds        = S[i+1]-S[i]
            st        = xq*ds 
            xs        = 3.0*PPX.c[0,i]*st**2 + 2.0*PPX.c[1,i]*st + PPX.c[2,i] 
            ys        = 3.0*PPY.c[0,i]*st**2 + 2.0*PPY.c[1,i]*st + PPY.c[2,i]
            sint      = np.dot(wq, np.sqrt(xs*xs + ys*ys))*ds
            serr      = max(serr, abs(sint-ds))
            Snew[i+1] = Snew[i] + sint
        
        S = Snew
        PPX = CubicSpline(S,X[0,:])    
        PPY = CubicSpline(S,X[1,:]) 

    return PPX, PPY

#-------------------------------------------------------------------------------
def splineval(PPX, PPY, S):
    # evaluates 2d spline at given S values
    # INPUT
    #   PP : two-dimensional spline structure 
    #   S  : arclength values at which to evaluate the spline
    # OUTPUT
    #   XY : coordinates on spline at the requested s values (2xN)

    XY       = np.concatenate((np.atleast_2d(PPX(S)),np.atleast_2d(PPY(S))) ,axis = 0)

    return XY


#-------------------------------------------------------------------------------
def splinetan(PPX,PPY, S):
    # evaluates 2d spline tangent (not normalized) at given S values
    # INPUT
    #   PP  : two-dimensional spline structure 
    #   S   : arclength values at which to evaluate the spline tangent
    # OUTPUT
    #   XYS : dX/dS and dY/dS values at each point (2xN)
 
    vec_1     = np.diag([3,2,1])
    vec_2     = np.zeros((1,3))
    C         = np.concatenate((np.atleast_2d(vec_1),np.atleast_2d(vec_2)) ,axis = 0)
    PPX_T     = PPX.c.T 
    PPX.c     = np.matmul(PPX_T,C).T
    PPX.order = 3
    PPY_T     = PPY.c.T 
    PPY.c     = np.matmul(PPY_T,C).T
    PPY.order = 3
    XYS       = np.concatenate((np.atleast_2d(PPX(S)),np.atleast_2d(PPY(S))) ,axis = 0) 
    return XYS



#-------------------------------------------------------------------------------
def splinetwo2one(PPX, PPY):
    # combines separate x,y splines into one 2d spline
    # INPUT
    #   PPX, PPY : one-dimensional spline structures
    # OUTPUT
    #   PP : two-dimensional spline structure 
    PP         = Data()
    PP.breaks  = PPX.x
    PP.xcoefs  = PPX.c.T
    PP.ycoefs  = PPY.c.T
    PP.pieces  = len(PPX.c[1])
    PP.order   = len(PPX.c[0])
    PP.dim     = 1

    return PP
#-------------------------------------------------------------------------------
def splineone2two(PP):
    # splits a 2d spline into two 1d splines
    # INPUT
    #   PP : two-dimensional spline structure
    # OUTPUT
    #   PPX, PPY : one-dimensional spline structures

    PPX = mkpp(PP.breaks, PP.xcoefs)
    PPY = mkpp(PP.breaks, PP.ycoefs) 
    return PPX, PPY

#-------------------------------------------------------------------------------
def quadseg():
    # Returns quadrature points and weights for a [0,1] line segment
    # INPUT
    # OUTPUT
    #   x : quadrature point coordinates (1d)
    #   w : quadrature weights

    x = np.array([ 0.046910077030668, 0.230765344947158, 0.500000000000000,\
        0.769234655052842, 0.953089922969332])
    w = np.array([  0.118463442528095, 0.239314335249683, 0.284444444444444,\
        0.239314335249683, 0.118463442528095])

    return x, w

def mkpp(breaks,coefs,*args):
    """
    Takes in the breaks, coefs, and optionally (d) then creates a pp from the 
    PiecePoly class and constructs the polynomial
    Returns: the constructed polynomial
    """
    if len(args)==1:
        d = np.transpose(args[0])
    else:
        d = 1
    sum=0
    try:
        #Just make sure coefs is not a 4D matrix
        for i in range(len(coefs)):
            for j in range(len(coefs[i])):
                sum = sum+len(coefs[i][j])
    except:
        #First try to count a 2D coefs array this should be the one that works
        try:
            for i in range(len(coefs)):
                sum = sum+len(coefs[i])
        #Coefs must be 1 dimensional
        except:
            sum = len(coefs)

    dlk = sum
    l = len(breaks)-1

    try:
        if len(d) > 1:
            prod = 0
            for i in range(len(d)):
                prod = prod*d[i]
            dl = prod*l
        else:
            dl = d*l
    except:
        dl = d*l

    k = dlk/dl+100*(math.pow(2,-52))
    if k<0:
        k = math.ceil(k)
    else:
        k = math.floor(k)

    if k<=0 or (dl*k!=dlk):
        print ("ERROR: MISMATCH PP AND COEF")
        return None
 
    pp        = PiecePoly()
    pp.form   = 'pp'
    pp.breaks = breaks 
    pp.coefs  = coefs
    pp.order  = k
    pp.dim    = d

    return pp


class PiecePoly():
    """
    A class to mimick the MATLAB struct piecewise polynomial (pp)
    """
    def __init__(self):
        self.form = 'pp'
        self.breaks = []
        self.coefs = []
        self.pieces = 0
        self.order = 0
        self.dim = 0
        
