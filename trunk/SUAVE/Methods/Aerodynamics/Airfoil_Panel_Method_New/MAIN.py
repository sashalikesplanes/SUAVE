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



from structures import * 
from paneling_functions import * 
from post_processing import * 
from supporting_functions import * 
from inviscid_functions import * 
from viscous_functions import * 


def main():
    # default mfoil object  

    airfoil_geometry       = ['NACA_2412.txt']  
    npanel                 = 199

    M       = Mfoil()
    M.geom  = struct_geom()     # geometry
    M.foil  = struct_panel()    # airfoil panels
    M.wake  = struct_panel()    # wake panels
    M.oper  = struct_oper()     # operating conditions
    M.isol  = struct_isol()     # inviscid solution variables
    M.vsol  = struct_vsol()     # viscous solution variables
    M.glob  = struct_glob()     # global system variables
    M.post  = struct_post()     # post-processing quantities
    M.param = struct_param()    # parameters    

    # set values  
    M.oper.alpha   = np.array([0])*Units.degrees
    M.oper.givencl = False
    M.oper.cltgt   = None  
    M.oper.Re      = np.array([1E5])
    M.oper.viscous = True 
    M.oper.Ma      = 0.1 



    make_panels(M, npanel)


    # component layout

    panels = Data()
    panels.pan_foil        = Data()
    panels.pan_foil.Layout = Data()
    panels.pan_foil.Layout.Row = 1     
    panels.pan_foil.Layout.Column = 1
    panels.pan_oper        = Data()
    panels.pan_oper.Layout = Data()
    panels.pan_oper.Layout.Row = 1    
    panels.pan_oper.Layout.Column = 2 
    panels.pan_param        = Data()
    panels.pan_param.Layout = Data() 
    panels.pan_param.Layout.Row = 2   
    panels.pan_param.Layout.Column = 1 
    panels.pan_plot        = Data()
    panels.pan_plot.Layout = Data()   
    panels.pan_plot.Layout.Row = 2     
    panels.pan_plot.Layout.Column = 2 
    panels.ax_plot        = Data()
    panels.ax_plot.Layout = Data()   
    panels.ax_plot.Layout.Row = [1,2]  
    panels.ax_plot.Layout.Column = 3 
    panels.ax_foil        = Data()
    panels.ax_foil.Layout = Data()   
    panels.ax_foil.Layout.Row = 3      
    panels.ax_foil.Layout.Column = 3 
    panels.ax_pan        = Data()
    panels.ax_pan.Layout = Data() 
    panels.ax_pan.Layout.Row = 3       
    panels.ax_pan.Layout.Column = [1,2]

    plot_panels(M)  
    #plot_test(M) 

    if (M.oper.viscous):
        solve_viscous(M) 
    else:
        solve_inviscid(M)    
    plot_cpplus(M)
    return 


def foil_name_set(M,txt):
    M.geom.name = txt.Value


def foil_naca_set(M,panels,txt): 
    M.geom_change('naca',txt.Value)
    pan_foil = panels[1] 
    gl_foil = pan_foil.Children[0] 
    foil_load_bt = gl_foil.Children(6) # load button
    foil_load_bt.Text = 'Load'
    #

    return 

def foil_npanel_set(M,panels,num):
    M.geom_panel(num.Value)
    #   

    return 


#-------------------------------------------------------------------------------
def foil_camb_press(M,panels,btn):
    file,path = uigetfile('*.*', 'Choose a plain-text coordinate file')
    ffile    = fullfile(path,file)
    X        = load(ffile) # attempt to load text file
    M.geom_change('coords',X)
    M.geom.name = file 
    btn.Text = file
    pan_foil = panels[0]  
    gl_foil = pan_foil.Children[0]  
    foil_naca_ef = gl_foil.Children[3]  # naca field
    foil_naca_ef.Value = ''
    #

    return
#-------------------------------------------------------------------------------
def foil_camb_press(M,panels,btn):
    file,path = uigetfile('*.*', 'Choose a plain-text camber file')
    ffile = fullfile(path,file)
    X = load(ffile) # attempt to load text file
    M.geom_addcamber(X)
    M.geom.name = [M.geom.name, ' camb']
    c#lear_solution(M,panels)
    return 

#-------------------------------------------------------------------------------
def foil_flap_press(M,panels,btn):
    pan_foil = panels[0]  
    gl_foil = pan_foil.Children[0] 
    xzhinge = [gl_foil.Children(10).Value, gl_foil.Children(11).Value]
    eta = gl_foil.Children(12).Value
    if (eta == 0):
        return  # nothing to do
    M.geom_flap(xzhinge, eta)
    M.geom.name = [M.geom.name, ' flap'] 
    return 


#-------------------------------------------------------------------------------
def set_xref_x(M,panels,num):
    M.geom.xref[0] = num.Value 
    return 


#-------------------------------------------------------------------------------
def set_xref_z(M,panels,num):
    M.geom.xref[1] = num.Value
    return 
 



#-------------------------------------------------------------------------------
def oper_initbl_press(M,btn):
    M.oper.initbl = ~M.oper.initbl
    if (M.oper.initbl) :
        btn.Text = 'Init BL' 
        btn.Value = False
    else:
        btn.Text = 'Reuse BL'
        btn.Value = True 
#-------------------------------------------------------------------------------
def set_conv_label(lbl, conv):
    if (conv):
        lbl.Text = 'Converged' 
        lbl.FontColor = 'g' 
        lbl.FontWeight = 'normal'
        lbl.Tooltip = 'Coupled solver converged'
    else:
        lbl.Text = 'NOT CONVERGED' 
        lbl.FontColor = 'r' 
        lbl.FontWeight = 'bold'
        lbl.Tooltip = 'Coupled solver did not converge try running an easier case, e.g. a lower alpha, and reusing the BL solution.' 
   



def Mfoil(): 
    
    M = Data()    
    M.geom  = struct_geom()     # geometry
    M.foil  = struct_panel()    # airfoil panels
    M.wake  = struct_panel()    # wake panels
    M.oper  = struct_oper()     # operating conditions
    M.isol  = struct_isol()     # inviscid solution variables
    M.vsol  = struct_vsol()     # viscous solution variables
    M.glob  = struct_glob()     # global system variables
    M.post  = struct_post()     # post-processing quantities
    M.param = struct_param()    # parameters  
    return M    

if __name__ == '__main__':  
    main()
    plt.show()