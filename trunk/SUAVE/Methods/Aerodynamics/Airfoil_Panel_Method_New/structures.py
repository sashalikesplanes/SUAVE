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

# =========================================================
# structures 
# =========================================================   
 
def struct_geom(): 
    S         = Data()
    S.chord   = 1         # chord length
    S.wakelen = 1.0     # wake extent length, in chords
    S.npoint  = 0        # number of geometry representation points 
    S.name    = 'NACA2412'   # airfoil name, e.g. NACA XXXX  
    S.xpoint  = np.array([[1,0.997515,0.993013785,0.987243446,0.98048,0.972887676,0.964580378,0.955643979,
                           0.946146748,0.936145,0.925686475,0.914812492,0.903559411,0.891959656,0.880042464,
                           0.867834443,0.85536,0.842641674,0.829700403,0.816555738,0.803226018,0.789728514,
                           0.776079552,0.762294611,0.748388414,0.734375,0.720267789,0.706079638,0.691822887,
                           0.677509406,0.663150627,0.648757585,0.634340941,0.619911012,0.605477795,0.591050985,
                           0.57664,0.562253995,0.547901877,0.533592324,0.519333796,0.505134543,0.491002626,
                           0.476945916,0.462972114,0.449088752,0.435303206,0.4216227,0.408054316,0.394605,
                           0.381281566,0.368090706,0.355038988,0.34213287,0.329378698,0.316782714,0.304351059,
                           0.292089774,0.280004811,0.268102027,0.256387198,0.24486601,0.233544074,0.222426918,
                           0.21152,0.200828701,0.190358333,0.18011414,0.1701013,0.160324927,0.150790073,0.14150173,
                           0.132464832,0.123684256,0.115164823,0.106911302,0.09892841,0.091220813,0.083793128,
                           0.076649923,0.069795721,0.063235,0.056972192,0.051011686,0.045357831,0.040014933,0.034987259,
                           0.030279037,0.025894455,0.021837666,0.018112787,0.014723896,0.01167504,0.00897023,0.006613444,
                           0.004608628,0.002959695,0.001670529,0.000744981,0.000186874,0,0.000186874,0.000744981,0.001670529,
                           0.002959695,0.004608628,0.006613444,0.00897023,0.01167504,0.014723896,0.018112787,0.021837666,
                           0.025894455,0.030279037,0.034987259,0.040014933,0.045357831,0.051011686,0.056972192,0.063235,
                           0.069795721,0.076649923,0.083793128,0.091220813,0.09892841,0.106911302,0.115164823,0.123684256,
                           0.132464832,0.14150173,0.150790073,0.160324927,0.1701013,0.18011414,0.190358333,0.200828701,0.21152,
                           0.222426918,0.233544074,0.24486601,0.256387198,0.268102027,0.280004811,0.292089774,0.304351059,
                           0.316782714,0.329378698,0.34213287,0.355038988,0.368090706,0.381281566,0.394605,0.408054316,0.4216227
                           ,0.435303206,0.449088752,0.462972114,0.476945916,0.491002626,0.505134543,0.519333796,0.533592324,
                           0.547901877,0.562253995,0.57664,0.591050985,0.605477795,0.619911012,0.634340941,0.648757585,
                           0.663150627,0.677509406,0.691822887,0.706079638,0.720267789,0.734375,0.748388414,0.762294611,
                           0.776079552,0.789728514,0.803226018,0.816555738,0.829700403,0.842641674,0.85536,0.867834443,
                           0.880042464,0.891959656,0.903559411,0.914812492,0.925686475,0.936145,0.946146748,0.955643979,
                           0.964580378,0.972887676,0.98048,0.987243446,0.993013785,0.997515,1],[-0.00126,-0.001442811,
                        -0.00177298,-0.002194463,-0.00268603,-0.003234807,-0.003831754,-0.004470018,-0.005144158,-0.00584972,
                        -0.006582982,-0.00734078,-0.008120391,-0.008919444,-0.009735854,-0.010567763,-0.011413505,-0.012271558
                        ,-0.013140522,-0.014019088,-0.014906012,-0.015800097,-0.016700176,-0.017605093,-0.018513691,
                        -0.019424799,-0.020337221,-0.02124973,-0.022161057,-0.023069886,-0.023974851,-0.024874532,
                        -0.02576745,-0.026652068,-0.027526793,-0.028389972,-0.029239897,-0.030074806,-0.030892889,-0.031692288
                        ,-0.032471106,-0.033227407,-0.033959227,-0.034664578,-0.035341455,-0.03598784,-0.036601716,-0.037181068
                        ,-0.037723891,-0.038230225,-0.038716379,-0.039184209,-0.039631063,-0.040054302,-0.040451304,-0.040819479
                        ,-0.041156269,-0.041459165,-0.041725711,-0.04195351,-0.042140232,-0.042283623,-0.042381506,-0.04243179
                        ,-0.042432472,-0.042381644,-0.042277493,-0.042118303,-0.041902463,-0.041628461,-0.041294888,-0.04090044
                        ,-0.040443911,-0.039924199,-0.039340299,-0.038691302,-0.03797639,-0.037194836,-0.036345995,-0.035429302
                        ,-0.034444268,-0.033390468,-0.03226754,-0.031075179,-0.029813126,-0.028481162,-0.027079103,-0.02560679
                        ,-0.024064081,-0.022450844,-0.02076695,-0.019012265,-0.017186639,-0.015289904,-0.013321863,-0.011282284
                        ,-0.009170894,-0.006987373,-0.004731348,-0.002402388,0,0.002439754,0.004880205,0.007320781,0.009760643
                        ,0.012198699,0.014633617,0.017063834,0.01948757,0.021902846,0.024307489,0.026699156,0.029075341,
                        0.031433392,0.033770528,0.03608385,0.038370359,0.040626968,0.042850521,0.045037801,0.047185551,
                        0.049290484,0.051349298,0.053358689,0.055315364,0.057216056,0.05905753,0.060836602,0.062550145,
                        0.064195101,0.065768491,0.067267426,0.06868911,0.070030855,0.071290085,0.072464343,0.073551295,
                        0.07454874,0.075454612,0.076266984,0.076984073,0.077604241,0.078126,0.078548011,0.078869089,
                        0.079088199,0.079204462,0.079217151,0.07912569,0.078929658,0.078628784,0.078222948,0.077716683
                        ,0.077129119,0.076463237,0.075720095,0.074900845,0.074006726,0.073039063,0.071999265,0.070888822,
                        0.069709299,0.068462337,0.067149655,0.065773042,0.064334363,0.062835557,0.06127864,0.059665708,
                        0.057998939,0.056280601,0.054513056,0.052698768,0.050840314,0.048940393,0.047001839,0.045027637,
                        0.043020939,0.040985084,0.038923617,0.03684032,0.034739234,0.032624696,0.030501374,0.028374312,
                        0.026248978,0.024131324,0.022027855,0.019945716,0.017892791,0.015877841,0.013910669,0.01200235,
                        0.010165548,0.008414976,0.006768108,0.00524636,0.003877256,0.002699052,0.001773458,0.00126]])
    #S.xpoint  = points       # point coordinates, [2 x npoint]
    S.xref    = np.array([0.25,0]) # moment reference point
    return S

#-------------------------------------------------------------------------------
def struct_panel():
    S = Data()
    S.N = 0            # number of nodes
    S.x = []           # node coordinates, [2 x N]
    S.s = []           # arclength values at nodes
    S.t = []           # dx/ds, dy/ds tangents at nodes
    return S

#-------------------------------------------------------------------------------
def struct_oper():

    S = Data()
    S.Vinf = 1          # velocity magnitude
    S.alpha = 0         # angle of attack, in degrees
    S.rho = 1           # density
    S.cltgt = 0         # lift coefficient target
    S.givencl = False   # true if cl is given instead of alpha
    S.initbl = True     # true to initialize the boundary layer
    S.viscous = False   # true to do viscous
    S.redowake = False  # true to rebuild wake after alpha changes
    S.Re = 1e5          # viscous Reynolds number
    S.Ma = 0            # Mach number
    return S

#-------------------------------------------------------------------------------
def struct_isol():
    S = Data()
    S.AIC = []          # aero influence coeff matrix
    S.gamref = []       # 0,90-deg alpha vortex strengths at airfoil nodes
    S.gam = []          # vortex strengths at airfoil nodes (for current alpha)
    S.sstag = 0.        # s location of stagnation point
    S.sstag_g = np.array([0,0])   # lin of sstag w.r.t. adjacent gammas
    S.sstag_ue = np.array([0,0])  # lin of sstag w.r.t. adjacent ue values
    S.Istag = np.array([0,0])    # node indices before/after stagnation
    S.sgnue = []        # +/- 1 on upper/lower surface nodes
    S.xi = []           # distance from the stagnation at all points
    S.uewi = []         # inviscid edge velocity in wake
    S.uewiref = []      # 0,90-deg alpha inviscid ue solutions on wake
    return S

#-------------------------------------------------------------------------------
def struct_vsol():
    S = Data()
    S.th  = []       # theta = momentum thickness [Nsys]
    S.ds = []        # delta star = displacement thickness [Nsys]
    S.Is = {}        # 3 cell arrays of surface indices
    S.wgap = []      # wake gap over wake points
    S.ue_m = []      # linearization of ue w.r.t. mass (all nodes)
    S.sigma_m = []   # d(source)/d(mass) matrix
    S.ue_sigma = []  # d(ue)/d(source) matrix
    S.turb = []      # flag over nodes indicating if turbulent (1) or lam (0) 
    S.xt = 0.        # transition location (xi) on current surface under consideration
    S.Xt = np.array([[0,0],[0,0]]) # transition xi/x for lower and upper surfaces 
    return S

#-------------------------------------------------------------------------------
def struct_glob():
    S = Data()
    S.Nsys = 0      # number of equations and states
    S.U = np.empty(shape=[1,0])     # primary states (th,ds,sa,ue) [4 x Nsys]
    S.dU = []       # primary state update
    S.dalpha = 0    # angle of attack update
    S.conv = True   # converged flag
    S.R = []        # residuals [3*Nsys x 1]
    S.R_U = []      # residual Jacobian w.r.t. primary states
    S.R_x = []      # residual Jacobian w.r.t. xi (s-values) [3*Nsys x Nsys]

    return S

#-------------------------------------------------------------------------------
def  struct_post():
    S = Data()
    S.cp = []       # cp distribution
    S.cpi = []      # inviscid cp distribution
    S.cl = 0        # lift coefficient
    S.cl_ue = []    # linearization of cl w.r.t. ue [N, airfoil only]
    S.cl_alpha = 0  # linearization of cl w.r.t. alpha
    S.cm = 0        # moment coefficient
    S.cdpi = 0      # near-field pressure drag coefficient
    S.cd = 0        # total drag coefficient
    S.cdf = 0       # skin friction drag coefficient
    S.cdp = 0       # pressure drag coefficient

    # distributions
    S.th = []       # theta = momentum thickness distribution
    S.ds = []       # delta* = displacement thickness distribution
    S.sa = []       # amplification factor/shear lag coeff distribution
    S.ue = []       # edge velocity (compressible) distribution
    S.uei = []      # inviscid edge velocity (compressible) distribution
    S.cf = []       # skin friction distribution
    S.Ret = []      # Re_theta distribution
    S.Hk = []       # kinematic shape parameter distribution

    return S

#-------------------------------------------------------------------------------
def struct_param():
    S = Data()

    S.verb   = 1     # printing verbosity level (higher -> more verbose)
    S.rtol   = 1e-10 # residual tolerance for Newton
    S.niglob = 50    # maximum number of global iterations
    S.doplot = True  # true to plot results after each solution
    S.axplot = []    # plotting axes (for more control of where plots go)

    # viscous parameters
    S.ncrit  = 9.0   # critical amplification factor    
    S.Cuq    = 1.0   # scales the uq term in the shear lag equation
    S.Dlr    = 0.9   # wall/wake dissipation length ratio
    S.SlagK  = 5.6   # shear lag constant

    # initial Ctau after transition
    S.CtauC  = 1.8   # Ctau constant
    S.CtauE  = 3.3   # Ctau exponent

    # G-beta locus: G = GA*sqrt(1+GB*beta) + GC/(H*Rt*sqrt(cf/2))
    S.GA     = 6.7   # G-beta A constant
    S.GB     = 0.75  # G-beta B constant
    S.GC     = 18.0  # G-beta C constant

    # operating conditions and thermodynamics
    S.Minf   = 0.    # freestream Mach number
    S.Vinf   = 0.    # freestream speed
    S.muinf  = 0.    # freestream dynamic viscosity
    S.mu0    = 0.    # stagnation dynamic viscosity
    S.rho0   = 1.    # stagnation density
    S.H0     = 0.    # stagnation enthalpy
    S.Tsrat  = 0.35  # Sutherland Ts/Tref
    S.gam    = 1.4   # ratio of specific heats
    S.KTb    = 1.    # Karman-Tsien beta = sqrt(1-Minf^2)
    S.KTl    = 0.    # Karman-Tsien lambda = Minf^2/(1+KTb)^2
    S.cps    = 0.    # sonic cp

    # station information
    S.simi   = False # true at a similarity station
    S.turb   = False # true at a turbulent station
    S.wake   = False # true at a wake station

    return S


