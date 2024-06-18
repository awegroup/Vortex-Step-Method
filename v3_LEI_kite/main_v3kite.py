# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 18:09:16 2022

@author: oriol2
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import functions_VSM_LLT as VSM



#%%  Input DATA 

#   Model and program specifics
ring_geo = '5fil' # System of vorticity defined with 5 or 3 filaments per wing section
model = 'VSM'     # Choose between Vortex Step Method (VSM) or Lifting Line Method (LLT)
# Convergence criteria
conv_crit = {
    'Niterations': 1000,
    'error' : 1e-5,
    'Relax_factor': 0.03
    }

# Wind speed vector definition  
Umag = 22               # Wind speed magnitude
aoa = 12*np.pi/180      # Angle of attack
ss = 0/180*np.pi  # ss angle
Uinf = np.array([np.cos(aoa)*np.cos(ss),np.sin(ss),np.sin(aoa)])*Umag # Wind speed vector

# Definition of the geometry
Atot = 19.753                    # Projected area 
CAD = VSM.get_CAD_matching_uri() # Geometry nodes locations
coord = VSM.struct2aero_geometry(CAD)/1000 # Change geometry to definition
N = int(len(coord)/2)            # Number of sections defined


# LE thickness at each section [m]
t = [0.118753,0.151561,0.178254,0.19406,0.202418,0.202418,0.19406,0.178254,0.151561,0.118753]

# Camber for each section (ct in my case)
# If you want it to vary do the same as the thickness
k = 0.095

# Plot thickness distribution
indexes = np.empty(N)
for i in range(N):
    indexes[i]=coord[2*i,1] 
fig = plt.figure(figsize = (6,5))
plt.plot(indexes,t,'ro',label='CAD Drawing')


# Number of splits per panel 
N_split = 4
# Refine structrural mesh into more panels
coord = VSM.refine_LEI_mesh(coord, N-1, N_split)
# Define system of vorticity
controlpoints,rings,bladepanels,ringvec,coord_L = VSM.create_geometry_LEI(
    coord, Uinf, int((N-1)*N_split+1),ring_geo,model)
N = int(len(coord)/2)           # Number of section after refining the mesh


#%% Airfoil Coefficients

# Definition of the thickness distribution for the refined mesh
thicc = np.array([])
for i in range(9):
    temp = np.linspace(t[i],t[i+1],N_split+1)
    temp1 = []
    for a in range(len(temp)-1):
        temp1.append((temp[a] +temp[a+1])/2)
    thicc = np.append(thicc,temp1)
#Plot thickness distribution
plt.plot(coord_L[:,1],thicc,color='blue',label='Fit')
plt.xlabel(r'$y$ [m]')
plt.ylabel(r'$t$ [m]')
plt.legend()

# Definition of airfoil coefficients 
# Based on Breukels (2011) correlation model 
aoas = np.arange(-20,21,1)
data_airf = np.empty((len(aoas),4,N-1))
t_c = np.empty(N-1)
for i in range(N-1):
    for j in range(len(aoas)):
        t_c[i] = thicc[i]/controlpoints[i]['chord']
        alpha = aoas[j]
        Cl,Cd,Cm = VSM.LEI_airf_coeff(t_c[i], k, alpha)
        data_airf[j,0,i] = alpha
        data_airf[j,1,i] = Cl
        data_airf[j,2,i] = Cd
        data_airf[j,3,i] = Cm
        
# Plot thickness ratio along the span
fig = plt.figure(figsize = (6,5))
plt.plot(coord_L[:,1],t_c)
plt.ylabel(r'Thickness ratio ($t/c$)')
plt.xlabel(r'$y$ [m]')

#%% SOLVER  

alphas = np.arange(-10,30,2)/180*np.pi #[10*np.pi/180] 
sideslips =[0/180*np.pi]#np.arange(0,15,2)/180*np.pi#

ss_map = []
aoa_map = []
CL = []
CD = []
CS = []
for aoa in alphas:
    for ss in sideslips:
        print(aoa,ss)
        Gamma0 = np.zeros(N-1)
        Uinf = np.array([np.cos(aoa)*np.cos(ss),np.sin(ss),np.sin(aoa)])*Umag # Wind speed vector
        # Define system of vorticity
        controlpoints,rings,bladepanels,ringvec,coord_L = VSM.create_geometry_LEI(
            coord, Uinf, N,ring_geo,model)
        # Solve for Gamma
        Fmag, Gamma,aero_coeffs = VSM.solve_lifting_line_system_matrix_approach_semiinfinite(
            ringvec, controlpoints, rings,Uinf,Gamma0,data_airf,conv_crit,model)
        #%OUTPUT Results
        F_rel,F_gl,Ltot,Dtot,CLi,CDi,CSi = VSM.output_results(Fmag,aero_coeffs,ringvec,Uinf,controlpoints,Atot)
        aoa_map.append(aoa)
        ss_map.append(ss)
        CL.append(CLi)
        CD.append(CDi)
        CS.append(CSi)
#%%
import pandas as pd
results = np.vstack((CL,CD,alphas*180/np.pi))
column_names = ['CL','CD','aoa']
df = pd.DataFrame(data=results.T, columns=column_names)

df.to_csv('VSM_data.csv', index=False)
#%%
CL_struts = np.loadtxt('RANS_CL_alpha_struts.csv',delimiter = ',') 
CD_struts = np.loadtxt('RANS_CD_alpha_struts.csv',delimiter = ',') 

CL_CFD = CL_struts[:,1]
aoas_CFD = CL_struts[:,0]
CD_CFD = np.interp(aoas_CFD,CD_struts[:,0],CD_struts[:,1])
results = np.vstack((CL_CFD,CD_CFD,aoas_CFD))
column_names = ['CL','CD','aoa']
df1 = pd.DataFrame(data=results.T, columns=column_names)
df1.to_csv('CFD_data.csv', index=False)
#%%

ss_map = np.array(ss_map)

CL = np.array(CL)
CD = np.array(CD)
CS = np.array(CS)
plt.figure()
plt.scatter(np.degrees(ss_map),CL**3/CD**2)
plt.plot
plt.figure()
plt.scatter(np.degrees(aoa_map),CL**3/CD**2)
plt.plot



# plt.figure()
# plt.plot(np.degrees(sideslips),CL**3/CD**2)
# plt.xlabel('ss')
# plt.ylabel('pow_factor')
# plt.grid()

# plt.figure()
# plt.plot(np.degrees(sideslips),CL)
# plt.xlabel('CL')
# plt.ylabel('ss')
# plt.grid()

# plt.figure()
# plt.plot(np.degrees(sideslips),CD)
# plt.xlabel('alpha')
# plt.ylabel('CD')
# plt.grid()

# plt.figure()
# plt.plot(np.degrees(sideslips),CL/CD)
# plt.xlabel('alpha')
# plt.ylabel('CL/CD')
# plt.grid()


CL = np.array(CL)
CD = np.array(CD)
CS = np.array(CS)

plt.figure()
plt.plot(np.degrees(alphas),CL**3/CD**2)
plt.xlabel('alpha')
plt.ylabel('pow_factor')
plt.grid()

plt.figure()
plt.plot(np.degrees(alphas),CL)
plt.xlabel('alpha')
plt.ylabel('CL')
plt.grid()

plt.figure()
plt.plot(np.degrees(alphas),CD)
plt.xlabel('alpha')
plt.ylabel('CD')
plt.grid()

plt.figure()
plt.plot(np.degrees(alphas),CL/CD)
plt.xlabel('alpha')
plt.ylabel('CL/CD')
plt.grid()


#%% PLOTS

fig = plt.figure(figsize = (6,5))
plt.plot(coord_L[:,1]/max(coord_L[:,1]),aero_coeffs[:,1], label = model) 
plt.xlabel(r'$y/s$')
plt.ylabel(r'$C_l$')
plt.legend()


fig = plt.figure(figsize = (6,5))
plt.plot(coord_L[:,1]/max(coord_L[:,1]),aero_coeffs[:,0]*180/np.pi, label = model) 
plt.xlabel(r'$y/s$')
plt.ylabel(r'$\alpha$')
plt.legend()


fig = plt.figure(figsize = (6,5))
plt.plot(coord_L[:,1]/max(coord_L[:,1]),aero_coeffs[:,2], label = model) 
plt.xlabel(r'$y/s$')
plt.ylabel(r'Viscous $C_d$')
plt.legend()


fig = plt.figure(figsize = (6,5))
plt.plot(coord_L[:,1]/max(coord_L[:,1]),Gamma, label = model) 
plt.xlabel(r'$y/s$')
plt.ylabel(r'$\Gamma$')
plt.legend()



#%% PLOT GEOMETRY

# xy plane
fig, ax = plt.subplots(figsize = (10,6))
for panel in bladepanels:
        coord = np.array([panel['p1'],panel['p2'],panel['p3'],panel['p4']])
        plt.plot(coord[:,1],coord[:,0],  'k',linestyle = '--')
        plt.plot(panel['p1'][1],panel['p1'][0],'o',mfc = 'white',c = 'k')
        plt.plot(panel['p4'][1],panel['p4'][0],'o',mfc = 'white',c = 'k')
for cp in controlpoints:       
    plt.plot(cp['coordinates'][1],cp['coordinates'][0], c = 'tab:green', marker = '*',markersize = 12,label = 'cp VSM')
    plt.plot(cp['coordinates_aoa'][1],cp['coordinates_aoa'][0],c = 'tab:red', marker = 'x', markersize = 12,label = 'cp LLT')  

# ax.set_ylim(0.8, -0.3)  # decreasing time
plt.grid(linestyle = ':', color = 'gray')

#zy plane
fig, ax = plt.subplots(figsize = (10,6))
for panel in bladepanels:
        coord = np.array([panel['p1'],panel['p2'],panel['p3'],panel['p4']])
        plt.plot(coord[:,1],coord[:,2],  'k',linestyle = '--')
        # plt.plot(panel['p1'][1],panel['p1'][2],'o',mfc = 'white',c = 'k')
        # plt.plot(panel['p4'][1],panel['p4'][2],'o',mfc = 'white',c = 'k')
for cp in controlpoints:       
    plt.plot(cp['coordinates'][1],cp['coordinates'][2], c = 'tab:green', marker = '*',markersize = 12,label = 'cp VSM')
    plt.plot(cp['coordinates_aoa'][1],cp['coordinates_aoa'][2],c = 'tab:red', marker = 'x', markersize = 12,label = 'cp LLT')  

# ax.set_ylim(0.8, -0.3)  # decreasing time
plt.grid(linestyle = ':', color = 'gray')
VSM.plot_geometry(bladepanels,controlpoints,rings,F_gl,coord_L,'True')