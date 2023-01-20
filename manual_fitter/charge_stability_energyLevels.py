# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 15:35:39 2022

@author: aivlev
"""

import numpy as np
import numpy.matlib
from matplotlib import pyplot as plt
from copy import deepcopy

def parse_manual_EC():
    manual_EC1_occ1 = [0.0, -1.0, -2.0, 0.0, 1.0, 2.0, -5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 0.0, 2.0, 3.0]
    manual_EC1_occ2 = [-1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    manual_EC1_val = [-0.013027747987717442, -0.011387595479884327, -0.013226128724096453, -0.013218488977290388, -0.013001588253252594, -0.01285473731160014, -0.015342250723899609, -0.013884805035114578, -0.013464219679981237, -0.013584219894873684, -0.011908648657416032, -0.012793643471182392, -0.013104136572093639, -0.012329685620854791, -0.015144454823027731]
    
    manual_EC1_occ1 = np.array(manual_EC1_occ1)
    manual_EC1_occ2 = np.array(manual_EC1_occ2)
    manual_EC1_val = np.array(manual_EC1_val)
    
    
    occ1_base = -5
    manual_EC1_occ1 = manual_EC1_occ1-occ1_base+1
    transitions_considered = manual_EC1_occ1>=0
    manual_EC1_occ1 = manual_EC1_occ1[transitions_considered]
    manual_EC1_occ2 = manual_EC1_occ2[transitions_considered]
    manual_EC1_val = manual_EC1_val[transitions_considered]
    manual_EC1_occ1_copy = deepcopy(manual_EC1_occ1)
    
    manual_EC1_val_parse = np.empty(int(max(manual_EC1_occ1)+1))
    
    i = 0
    while len(manual_EC1_occ1_copy)>0:
        same_transition = (manual_EC1_occ1_copy == i)
        manual_EC1_val_parse[i] = np.mean(manual_EC1_val[same_transition])
        
        manual_EC1_occ1_copy = manual_EC1_occ1_copy[np.logical_not(same_transition)]
        manual_EC1_val = manual_EC1_val[np.logical_not(same_transition)]
        i += 1
    manual_EC1_val_parse *= -1000
    manual_EC1_val_parse[0] = 0
    return manual_EC1_val_parse


def make_energy_levels_test(EC2 = 0.8,n_1=5,n_2=5,N_gateX=100,N_gateY=200):  
    
    occupation_1 = np.arange(n_1)
    occupation_2 = np.arange(n_2)
    

    template_X = np.ones(N_gateX)
    template_Y = np.ones(N_gateY)
        
    occupation_1_grid, occupation_2_grid,_,_  = np.meshgrid(occupation_1,occupation_2,template_X,template_Y,indexing='ij')
    
    manual_EC1 = True
    if manual_EC1:
        EC1 = 1.0
        EC1 = EC1*np.ones(n_1)
        
        EC1[3]=0.6
        EC1[4]=1.4
    else:
        EC1 = 1.0
        EC1 = EC1*np.ones(n_1)

    EC1 = np.tile(EC1,(n_2,1)).T
    EC1 = np.cumsum(EC1,axis=0)
    
    
    EC12_base = 0.3*np.ones((n_1,n_2))
    EC12_base[0,:] = 0
    EC12_base[:,0] = 0
    manual_EC12 = True
    
    if manual_EC12:
        EC12_base[:,2:] = 0.2
        EC12_base[:,3:] = 0.15
        EC12_base[:,4:] = 0.07

    else:
        pass
    EC12_base_1 = np.cumsum(EC12_base,axis=1)
    EC12_base_2 = np.cumsum(EC12_base,axis=0)
    
    EC12_occ_var2 = 0.0

    manual_EC2 = True
    EC2_old = deepcopy(EC2)
    if manual_EC2 == True:
        EC2[:,3] = 0.3
        EC2[:,4] = -0.2
        EC2[:,5] = 0.1
        EC2[:,6] = 0.3
    else:
        pass

    EC2 = np.cumsum(EC2,axis=1)
    
    mu_1 = np.empty(np.shape(occupation_1_grid))
    mu_2 = np.empty(np.shape(occupation_1_grid))
    
    
    for i in occupation_1:
        for j in occupation_2:

            EC12_1 = EC12_base_1[i,j]+(j-1)*(j>0)*EC12_occ_var2
            EC12_2 = EC12_base_2[i,j]+(j-1)*(j>0)*EC12_occ_var2
    
            # new_EC1 = EC1[i]*occupation_1_grid[i,j,:,:]
            mu_1[i,j,:,:] = EC1[i,j]+EC12_1-EC1[1,j]
            mu_2[i,j,:,:] = EC2[i,j]+EC12_2-EC2[i,1]
    
    
    mu_1[-1,:] = 1000 # to preventing running out of matrix indices later, without resorting to if-statements
    mu_1[:,-1] = 1000
    
    mu_2[-1,:] = 1000 # to preventing running out of matrix indices later, without resorting to if-statements
    mu_2[:,-1] = 1000

    print(mu_2[0,:,0,0])    

    return mu_1, mu_2

def make_energy_levels_BL_new(EC2 = 0.8,n_1=5,n_2=5,N_gateX=100,N_gateY=200):  
    
    occupation_1 = np.arange(n_1)
    occupation_2 = np.arange(n_2)
    

    template_X = np.ones(N_gateX)
    template_Y = np.ones(N_gateY)
        
    occupation_1_grid, occupation_2_grid,_,_  = np.meshgrid(occupation_1,occupation_2,template_X,template_Y,indexing='ij')
    
    manual_EC1 = True
    if manual_EC1:
        manual_EC1_val = parse_manual_EC()

        EC1 = manual_EC1_val/13
        added_length = n_1-len(EC1)
        if added_length>0:
            added_array = np.ones(added_length)
            EC1 = np.concatenate((EC1,added_array))
        
        EC1[2] = 1.08
        EC1[3] = 1
        EC1[4] = 1.0
        EC1[5] = 1.04
        # EC1[5] = 1.09
        EC1[6] = 1
        EC1[7] = 1
        EC1[8] = 1
        EC1[9] = 1.05
        EC1[10] = 1.1
        EC1[11:] = 0.8
        EC1[12:] = 0.8
        # EC1[4] = 0.2
    else:
        EC1 = 1.0
        EC1 = EC1*np.ones(n_1)
        # EC1[4] = 0.9
    # EC1 = 1.0
    # EC2 = 1.0
    # print(EC1)
    EC1 = np.tile(EC1,(n_2,1)).T
    # EC1[5,1:] = 1.09
    EC1 = np.cumsum(EC1,axis=0)
    
    
    EC12_base = 0.8*np.ones((n_1,n_2))
    EC12_base[0,:] = 0
    EC12_base[:,0] = 0
    manual_EC12 = True
    
    if manual_EC12:
        EC12_base[2,1:] = 0.75
        EC12_base[3,1:] = 0.75
        EC12_base[4,1:] = 0.76
        EC12_base[5,1:] = 0.79
        EC12_base[6,1:] = 0.75
        EC12_base[7,1:] = 0.65
        EC12_base[8,1:] = 0.62
        EC12_base[9,1:] = 0.604
        EC12_base[10,1:] = 0.604
        EC12_base[11,1:] = 0.604
        EC12_base[12,1:] = 0.604
        
        EC12_base[:,2] = EC2[0,2]/EC2[0,1]*EC12_base[:,1]
        
        EC12_base[1,2:] = 0.5
        EC12_base[2,2:] = 0.52
        EC12_base[3,2:] = 0.51
        EC12_base[4,2:] = 0.48
        EC12_base[5,2:] = 0.48
        EC12_base[6,2:] = 0.5
        EC12_base[7,2:] = 0.5
        EC12_base[8,2:] = 0.48
        EC12_base[9,2:] = 0.45
        EC12_base[10,2:] = 0.50
        EC12_base[11,2:] = 0.50
        EC12_base[12,2:] = 0.50
        
        EC12_base[1,3:] = 0.0
        EC12_base[2,3:] = 0.2
        EC12_base[3,3:] = 0.2
        EC12_base[4,3:] = 0.25
        EC12_base[5,3:] = 0.17
        EC12_base[6,3:] = 0.22
        EC12_base[7,3:] = 0.2
        EC12_base[8,3:] = 0.17
        EC12_base[9,3:] = 0.2     
        EC12_base[10,3:] = 0.2   
        EC12_base[11,3:] = 0.2   
        EC12_base[12,3:] = 0.2   

        EC12_base[4,4:] = 0.35
        EC12_base[5,4:] = 0.35
        EC12_base[6,4:] = 0.3
        EC12_base[7,4:] = 0.25
        EC12_base[8,4:] = 0.25
        EC12_base[9,4:] = 0.27     
        EC12_base[10,4:] = 0.25   
        EC12_base[11,4:] = 0.2   
        EC12_base[12,4:] = 0.2   


    else:
        pass
    EC12_base_1 = np.cumsum(EC12_base,axis=1)
    EC12_base_2 = np.cumsum(EC12_base,axis=0)
    
    EC12_occ_var2 = 0.0

    manual_EC2 = True
    EC2_old = deepcopy(EC2)
    if manual_EC2 == True:
        EC2[:,2] = 0.9
        EC2[:,3] = 0.05
        EC2[:,4:] = 1.2
        EC2[:,5:] = 0.5
        pass
    else:
        pass
    print(EC2[0,:])
    print(EC2_old[0,:])
    interdot = (EC2_old[0,:]/17.5*13)
    print(interdot)
    EC2_norm = EC2[0,:]/interdot
    print(EC2_norm)
    EC2 = np.cumsum(EC2,axis=1)
    
    mu_1 = np.empty(np.shape(occupation_1_grid))
    mu_2 = np.empty(np.shape(occupation_1_grid))
    
    
    for i in occupation_1:
        for j in occupation_2:

            EC12_1 = EC12_base_1[i,j]+(j-1)*(j>0)*EC12_occ_var2
            EC12_2 = EC12_base_2[i,j]+(j-1)*(j>0)*EC12_occ_var2
    
            # new_EC1 = EC1[i]*occupation_1_grid[i,j,:,:]
            mu_1[i,j,:,:] = EC1[i,j]+EC12_1-EC1[1,j]
            mu_2[i,j,:,:] = EC2[i,j]+EC12_2-EC2[i,1]
    
    
    mu_1[-1,:] = 1000 # to preventing running out of matrix indices later, without resorting to if-statements
    mu_1[:,-1] = 1000
    
    mu_2[-1,:] = 1000 # to preventing running out of matrix indices later, without resorting to if-statements
    mu_2[:,-1] = 1000

    print(mu_2[0,:,0,0])    

    return mu_1, mu_2

def make_energy_levels_BL(EC2 = 0.8,n_1=5,n_2=5,N_gateX=100,N_gateY=200):  
    
    occupation_1 = np.arange(n_1)
    occupation_2 = np.arange(n_2)
    

    template_X = np.ones(N_gateX)
    template_Y = np.ones(N_gateY)
        
    occupation_1_grid, occupation_2_grid,_,_  = np.meshgrid(occupation_1,occupation_2,template_X,template_Y,indexing='ij')
    
    manual_EC1 = True
    if manual_EC1:
        manual_EC1_val = parse_manual_EC()

        EC1 = manual_EC1_val/13
        added_length = n_1-len(EC1)
        if added_length>0:
            added_array = np.ones(added_length)
            EC1 = np.concatenate((EC1,added_array))
        
        EC1[2] = 1.11
        EC1[3] = 1
        EC1[4] = 1.06
        EC1[5] = 1.04
        # EC1[5] = 1.09
        EC1[6] = 0.85
        EC1[7] = 1
        EC1[8] = 1.05
        EC1[9] = 1
        EC1[10] = 1.3
        EC1[11:] = 0.8
        EC1[12:] = 1
        # EC1[4] = 0.2
    else:
        EC1 = 1.0
        EC1 = EC1*np.ones(n_1)
        # EC1[4] = 0.9
    # EC1 = 1.0
    # EC2 = 1.0
    print(EC1)
    EC1 = np.tile(EC1,(n_2,1)).T
    # EC1[5,1:] = 1.09
    EC1 = np.cumsum(EC1,axis=0)
    
    
    EC12_base = 0.5*np.ones((n_1,n_2))
    EC12_base[0,:] = 0
    EC12_base[:,0] = 0
    manual_EC12 = True
    
    if manual_EC12:
        EC12_base[2,1] = 0.56
        EC12_base[3,1] = 0.65
        EC12_base[4,1] = 0.6
        # EC12_base[5,1] = 0.63
        EC12_base[5,1] = 0.62
        EC12_base[6,1] = 0.62
        EC12_base[7,1] = 0.65
        EC12_base[8,1] = 0.62
        EC12_base[9,1] = 0.604
        
        EC12_base[:,2] = EC2[0,2]/EC2[0,1]*EC12_base[:,1]
        
        EC12_base[1,2] = 0.35
        EC12_base[2,2] = 0.3
        EC12_base[3,2] = 0.28
        EC12_base[4,2] = 0.3
        EC12_base[5,2] = 0.33
        EC12_base[6,2] = 0.4
        EC12_base[7,2] = 0.38
        EC12_base[8,2] = 0.4
        EC12_base[9,2] = 0.38
        EC12_base[10,2] = 0.38
        
        EC12_base[:,3] = EC2[0,3]/EC2[0,2]*EC12_base[:,2]
        
        EC12_base[1,3:] = 0.1
        EC12_base[2,3:] = 0
        EC12_base[3,3:] = 0.04
        EC12_base[4,3:] = 0.1
        EC12_base[5,3:] = 0.22
        EC12_base[6,3:] = 0.27
        EC12_base[7,3:] = 0.2
        EC12_base[8,3:] = 0.19
        EC12_base[9,3:] = 0.21
        EC12_base[10,3:] = 0.21
        
        
        EC12_base[1,4:] = 0.08
        EC12_base[2,4:] = 0.0
        EC12_base[3,4:] = 0.0
        EC12_base[4,4:] = 0.25
        EC12_base[5,4:] = 0.2
        EC12_base[6,4:] = 0.2
        EC12_base[7,4:] = 0.23
        EC12_base[8,4:] = 0.19
        EC12_base[9,4:] = 0.21
        EC12_base[10,4:] = 0.21
        EC12_base[11:,4:] = 0.15
        
        # EC12_base[:,5] = EC2[0,5]/EC2[0,4]*EC12_base[:,4]
        
        EC12_base[1,5:] = 0.0
        EC12_base[2,5:] = 0.0
        EC12_base[3,5:] = 0.15
        EC12_base[4,5:] = 0.15
        EC12_base[5,5:] = 0.15
        EC12_base[6,5:] = 0.2
        EC12_base[7,5:] = 0.2
        EC12_base[8,5:] = 0.15
        EC12_base[9,5:] = 0.17
        EC12_base[10,5:] = 0.21
        
        EC12_base[11:,5:] = 0.15
        

    else:
        pass
    EC12_base_1 = np.cumsum(EC12_base,axis=1)
    EC12_base_2 = np.cumsum(EC12_base,axis=0)
    
    EC12_occ_var2 = 0.0
    EC2_old = deepcopy(EC2)
    manual_EC2 = True
    if manual_EC2 == True:
        EC2[:,3] = 0.25
        EC2[:,4:] = 0.15
        EC2[:,5:] = 0.0
        pass
    else:
        pass
    print(EC2[0,:])
    print(EC2_old[0,:])
    interdot = (EC2_old[0,:]/17.5*13)
    print(interdot)
    EC2_norm = EC2[0,:]/interdot
    print(EC2_norm)
    EC2 = np.cumsum(EC2,axis=1)
    
    mu_1 = np.empty(np.shape(occupation_1_grid))
    mu_2 = np.empty(np.shape(occupation_1_grid))
    
    
    for i in occupation_1:
        for j in occupation_2:

            EC12_1 = EC12_base_1[i,j]+(j-1)*(j>0)*EC12_occ_var2
            EC12_2 = EC12_base_2[i,j]+(j-1)*(j>0)*EC12_occ_var2
    
            # new_EC1 = EC1[i]*occupation_1_grid[i,j,:,:]
            mu_1[i,j,:,:] = EC1[i,j]+EC12_1-EC1[1,j]
            mu_2[i,j,:,:] = EC2[i,j]+EC12_2-EC2[i,1]
    
    
    mu_1[-1,:] = 1000 # to preventing running out of matrix indices later, without resorting to if-statements
    mu_1[:,-1] = 1000
    
    mu_2[-1,:] = 1000 # to preventing running out of matrix indices later, without resorting to if-statements
    mu_2[:,-1] = 1000

    print(mu_2[0,:,0,0])    

    return mu_1, mu_2

def make_energy_levels_SSL(EC2 = 0.8,n_1=5,n_2=5,N_gateX=100,N_gateY=200):  
    
    occupation_1 = np.arange(n_1)
    occupation_2 = np.arange(n_2)
    

    template_X = np.ones(N_gateX)
    template_Y = np.ones(N_gateY)
        
    occupation_1_grid, occupation_2_grid,_,_  = np.meshgrid(occupation_1,occupation_2,template_X,template_Y,indexing='ij')
    
    manual_EC1 = True
    if manual_EC1:
        manual_EC1_val = parse_manual_EC()

        EC1 = manual_EC1_val/13
        added_length = n_1-len(EC1)
        if added_length>0:
            added_array = np.ones(added_length)
            EC1 = np.concatenate((EC1,added_array))
        
        EC1[2] = 1.15
        EC1[3] = 1.12
        EC1[4] = 1.1
        EC1[5] = 1.00
        # EC1[5] = 1.09
        EC1[6] = 1.1
        EC1[7] = 1.0
        EC1[8] = 1.15
        EC1[10] = 1.15

    else:
        EC1 = 1.0
        EC1 = EC1*np.ones(n_1)
        # EC1[4] = 0.9
    # EC1 = 1.0
    # EC2 = 1.0
    print(EC1)
    EC1 = np.tile(EC1,(n_2,1)).T
    # EC1[5,1:] = 1.09
    EC1 = np.cumsum(EC1,axis=0)
    
    
    EC12_base = 0.5*np.ones((n_1,n_2))
    EC12_base[0,:] = 0
    EC12_base[:,0] = 0
    
    manual_EC12 = True
    
    if manual_EC12:
        EC12_base[1,1:] = 0.8
        EC12_base[2,1:] = 0.8
        EC12_base[3,1:] = 0.7
        EC12_base[4,1:] = 0.7
        EC12_base[5,1:] = 0.8
        EC12_base[6,1:] = 0.7
        EC12_base[7,1:] = 0.7
        EC12_base[8,1:] = 0.7
        
        EC12_base[1,2:] = 0.3
        EC12_base[2,2:] = 0.45
        EC12_base[3,2:] = 0.4
        EC12_base[4,2:] = 0.5
        EC12_base[5,2:] = 0.4
        EC12_base[6,2:] = 0.4
        EC12_base[7,2:] = 0.4
        EC12_base[8,2:] = 0.4
        
        EC12_base[1,3:] = 0.15
        EC12_base[2,3:] = 0.2
        EC12_base[3,3:] = 0.25
        EC12_base[4,3:] = 0.25
        EC12_base[5,3:] = 0.25
        EC12_base[6,3:] = 0.2
        EC12_base[7,3:] = 0.2
        EC12_base[8,3:] = 0.2
        EC12_base[9,3:] = 0.3
        EC12_base[10,3:] = 0.3
    else:
        pass
    
    EC12_base_1 = np.cumsum(EC12_base,axis=1)
    EC12_base_2 = np.cumsum(EC12_base,axis=0)
    
    EC12_occ_var2 = 0.0

    EC2_old = deepcopy(EC2)
    manual_EC2 = True
    if manual_EC2 == True:
        EC2[:,2] = 0.9
        EC2[:,3] = 0.3
        EC2[:,4:] = 0.25
        EC2[:,5:] = 0.0
        pass
    else:
        pass
    print(EC2[0,:])
    print(EC2_old[0,:])
    interdot = (EC2_old[0,:]/17.5*13)
    print(interdot)
    EC2_norm = EC2[0,:]/interdot
    print(EC2_norm)
    EC2 = np.cumsum(EC2,axis=1)
    
    mu_1 = np.empty(np.shape(occupation_1_grid))
    mu_2 = np.empty(np.shape(occupation_1_grid))
    
    
    for i in occupation_1:
        for j in occupation_2:

            EC12_1 = EC12_base_1[i,j]+(j-1)*(j>0)*EC12_occ_var2
            EC12_2 = EC12_base_2[i,j]+(j-1)*(j>0)*EC12_occ_var2
    
            # new_EC1 = EC1[i]*occupation_1_grid[i,j,:,:]
            mu_1[i,j,:,:] = EC1[i,j]+EC12_1-EC1[1,j]
            mu_2[i,j,:,:] = EC2[i,j]+EC12_2-EC2[i,1]
    
    
    mu_1[-1,:] = 1000 # to preventing running out of matrix indices later, without resorting to if-statements
    mu_1[:,-1] = 1000
    
    mu_2[-1,:] = 1000 # to preventing running out of matrix indices later, without resorting to if-statements
    mu_2[:,-1] = 1000

    print(mu_2[0,:,0,0])    

    return mu_1, mu_2

def get_detunings_test(n_1=5,n_2=5,N_gateX=100,N_gateY=200,min_x=-10,max_x=5,min_y=-10,max_y=5):
    
    template_1 = np.ones(n_1)
    template_2 = np.ones(n_2)

    
    gate_X = np.linspace(min_x,max_x,N_gateX)
    gate_Y = np.linspace(min_y,max_y,N_gateY)
    
    
    _,_,gate_X_grid, gate_Y_grid = np.meshgrid(template_1,template_2,gate_X,gate_Y,indexing='ij')
    
    detuning_1 = np.empty(np.shape(gate_X_grid))
    detuning_2 = np.empty(np.shape(gate_X_grid))
    
    
    mV_per_EC1 = 13 #mV/EC1 #13
    mV_per_EC2 = 17.5 #14
    
    slope_1_base = -2.5
    slope_2_base = -1.2
    
    slope_2_min = -0.1
    
    slope_occ_var_22 = 0.19
    slope_gate_var_2Y = 0e-2
    
    slope_interdot_base = -6
    slope_interdot_base = -6*np.ones((n_1,n_2))
    slope_interdot_base_manual = True
    if slope_interdot_base_manual:
        slope_interdot_base[:,2] = -5
        slope_interdot_base[:,3:] = -4
        slope_interdot_base[:,4:] = -3
        # slope_interdot_base[:,5] = -3
        slope_occ_var_i2 = 0
    else:
        slope_occ_var_i2 = 2
    
    EC2 = np.empty((n_1,n_2))
    
    for i in range(np.shape(gate_X_grid)[0]):
        for j in range(np.shape(gate_X_grid)[1]):
            
            slope_1 = slope_1_base
            slope_2_epsilon0 = min(slope_2_base+slope_occ_var_22*(j-1)*(j>0),slope_2_min)
            slope_2 = slope_2_epsilon0-slope_gate_var_2Y*gate_Y_grid[i,j,:,:]
            
            slope_interdot = slope_interdot_base[i,j]+slope_occ_var_i2*(j-1)*(j>0)
            
            inter_ratio = (1+slope_interdot*(-1/slope_1))/(1+slope_interdot*(-1/slope_2))
            
            detuning_1[i,j,:,:] = gate_X_grid[i,j,:,:]-1/(slope_1)*gate_Y_grid[i,j,:,:]
            detuning_2[i,j,:,:] = gate_X_grid[i,j,:,:]-1/(slope_2)*gate_Y_grid[i,j,:,:]
            detuning_2[i,j,:,:] *= inter_ratio
            
            detuning_1[i,j,:,:] /= mV_per_EC1
            detuning_2[i,j,:,:] /= mV_per_EC1
            
            EC2[i,j] = (1+slope_interdot*(-1/slope_1))/(1+slope_interdot*(-1/slope_2_epsilon0))*mV_per_EC2/mV_per_EC1

    
    detuning_1 *= -1
    detuning_2 *= -1
    

    # detuning_1 = gate_X_grid-1/(slope_1)*gate_Y_grid
    
    # detuning_2 = inter_ratio*(gate_X_grid-1/(slope_2)*gate_Y_grid)
    # detuning_2 = -detuning_2
    
    return detuning_1,detuning_2,EC2


def get_detunings_BL_new(n_1=5,n_2=5,N_gateX=100,N_gateY=200,min_x=-10,max_x=5,min_y=-10,max_y=5):
    
    template_1 = np.ones(n_1)
    template_2 = np.ones(n_2)

    
    gate_X = np.linspace(min_x,max_x,N_gateX)
    gate_Y = np.linspace(min_y,max_y,N_gateY)
    
    
    _,_,gate_X_grid, gate_Y_grid = np.meshgrid(template_1,template_2,gate_X,gate_Y,indexing='ij')
    
    detuning_1 = np.empty(np.shape(gate_X_grid))
    detuning_2 = np.empty(np.shape(gate_X_grid))
    
    
    mV_per_EC1 = 13 #mV/EC1 #13
    mV_per_EC2 = 17.5 #14
    
    slope_1_base = -2.1
    slope_2_base = -1.85
    
    slope_2_min = -1.15
    
    slope_occ_var_22 = 0.15
    slope_gate_var_2Y = 0e-2
    
    slope_interdot_base = -6
    slope_interdot_base = -6*np.ones((n_1,n_2))
    slope_interdot_base_manual = True
    if slope_interdot_base_manual:
        slope_interdot_base[:,2] = -3.3
        slope_interdot_base[:,3:] = -2.4
        slope_interdot_base[:,4:] = -3
        # slope_interdot_base[:,5] = -3
        slope_occ_var_i2 = 0
    else:
        slope_occ_var_i2 = 2
    
    EC2 = np.empty((n_1,n_2))
    
    for i in range(np.shape(gate_X_grid)[0]):
        for j in range(np.shape(gate_X_grid)[1]):
            
            slope_1 = slope_1_base
            slope_2_epsilon0 = min(slope_2_base+slope_occ_var_22*(j-1)*(j>0),slope_2_min)
            slope_2 = slope_2_epsilon0-slope_gate_var_2Y*gate_Y_grid[i,j,:,:]
            
            slope_interdot = slope_interdot_base[i,j]+slope_occ_var_i2*(j-1)*(j>0)
            
            inter_ratio = (1+slope_interdot*(-1/slope_1))/(1+slope_interdot*(-1/slope_2))
            
            detuning_1[i,j,:,:] = gate_X_grid[i,j,:,:]-1/(slope_1)*gate_Y_grid[i,j,:,:]
            detuning_2[i,j,:,:] = gate_X_grid[i,j,:,:]-1/(slope_2)*gate_Y_grid[i,j,:,:]
            detuning_2[i,j,:,:] *= inter_ratio
            
            detuning_1[i,j,:,:] /= mV_per_EC1
            detuning_2[i,j,:,:] /= mV_per_EC1
            
            EC2[i,j] = (1+slope_interdot*(-1/slope_1))/(1+slope_interdot*(-1/slope_2_epsilon0))*mV_per_EC2/mV_per_EC1

    
    detuning_1 *= -1
    detuning_2 *= -1
    

    # detuning_1 = gate_X_grid-1/(slope_1)*gate_Y_grid
    
    # detuning_2 = inter_ratio*(gate_X_grid-1/(slope_2)*gate_Y_grid)
    # detuning_2 = -detuning_2
    
    return detuning_1,detuning_2,EC2

def get_detunings_BL(n_1=5,n_2=5,N_gateX=100,N_gateY=200,min_x=-10,max_x=5,min_y=-10,max_y=5):
    
    template_1 = np.ones(n_1)
    template_2 = np.ones(n_2)

    
    gate_X = np.linspace(min_x,max_x,N_gateX)
    gate_Y = np.linspace(min_y,max_y,N_gateY)
    
    
    _,_,gate_X_grid, gate_Y_grid = np.meshgrid(template_1,template_2,gate_X,gate_Y,indexing='ij')
    
    detuning_1 = np.empty(np.shape(gate_X_grid))
    detuning_2 = np.empty(np.shape(gate_X_grid))
    
    
    mV_per_EC1 = 13 #mV/EC1 #13
    mV_per_EC2 = 17.5 #14
    
    slope_1_base = -2.2
    slope_2_base = -1.8
    
    slope_2_min = -1.15
    
    slope_occ_var_22 = 0.3
    slope_gate_var_2Y = 0e-2
    
    slope_interdot_base = -6
    slope_interdot_base = -6*np.ones((n_1,n_2))
    slope_interdot_base_manual = True
    if slope_interdot_base_manual:
        slope_interdot_base[:,2] = -4
        slope_interdot_base[:,3:] = -3.3
        # slope_interdot_base[:,4:] = -3
        slope_interdot_base[:,5] = -3
        slope_occ_var_i2 = 0
    else:
        slope_occ_var_i2 = 2
    
    EC2 = np.empty((n_1,n_2))
    
    for i in range(np.shape(gate_X_grid)[0]):
        for j in range(np.shape(gate_X_grid)[1]):
            
            slope_1 = slope_1_base
            slope_2_epsilon0 = min(slope_2_base+slope_occ_var_22*(j-1)*(j>0),slope_2_min)
            slope_2 = slope_2_epsilon0-slope_gate_var_2Y*gate_Y_grid[i,j,:,:]
            
            slope_interdot = slope_interdot_base[i,j]+slope_occ_var_i2*(j-1)*(j>0)
            
            inter_ratio = (1+slope_interdot*(-1/slope_1))/(1+slope_interdot*(-1/slope_2))
            
            detuning_1[i,j,:,:] = gate_X_grid[i,j,:,:]-1/(slope_1)*gate_Y_grid[i,j,:,:]
            detuning_2[i,j,:,:] = gate_X_grid[i,j,:,:]-1/(slope_2)*gate_Y_grid[i,j,:,:]
            detuning_2[i,j,:,:] *= inter_ratio
            
            detuning_1[i,j,:,:] /= mV_per_EC1
            detuning_2[i,j,:,:] /= mV_per_EC1
            
            EC2[i,j] = (1+slope_interdot*(-1/slope_1))/(1+slope_interdot*(-1/slope_2_epsilon0))*mV_per_EC2/mV_per_EC1

    
    detuning_1 *= -1
    detuning_2 *= -1
    

    # detuning_1 = gate_X_grid-1/(slope_1)*gate_Y_grid
    
    # detuning_2 = inter_ratio*(gate_X_grid-1/(slope_2)*gate_Y_grid)
    # detuning_2 = -detuning_2
    
    return detuning_1,detuning_2,EC2

def get_detunings_SSL(n_1=5,n_2=5,N_gateX=100,N_gateY=200,min_x=-10,max_x=5,min_y=-10,max_y=5):
    
    template_1 = np.ones(n_1)
    template_2 = np.ones(n_2)

    
    gate_X = np.linspace(min_x,max_x,N_gateX)
    gate_Y = np.linspace(min_y,max_y,N_gateY)
    
    
    _,_,gate_X_grid, gate_Y_grid = np.meshgrid(template_1,template_2,gate_X,gate_Y,indexing='ij')
    
    detuning_1 = np.empty(np.shape(gate_X_grid))
    detuning_2 = np.empty(np.shape(gate_X_grid))
    
    
    mV_per_EC1 = 13 #mV/EC1 #13
    mV_per_EC2 = 17.5 #14
    
    slope_1_base = -1.35
    # slope_occ_var_11 = 0
    slope_gate_var_1X = -2.5e-3 #-1e-3
    
    
    slope_2_base = -1.2
    slope_2_max = -0.7
    slope_occ_var_22 = 0.3
    slope_gate_var_2X = -1.6e-3 #-2e-3
    
    slope_interdot_base = -5
    slope_interdot_base = slope_interdot_base*np.ones((n_1,n_2))
    slope_interdot_base_manual = False
    if slope_interdot_base_manual:
        slope_interdot_base[:,2] = -2
        # slope_interdot_base[:,3:] = -3.3
        # # slope_interdot_base[:,4:] = -3
        # slope_interdot_base[:,5] = -3
        # slope_occ_var_i2 = 0
        pass
    else:
        slope_occ_var_i2 = 1
    
    EC2 = np.empty((n_1,n_2))
    
    for i in range(np.shape(gate_X_grid)[0]):
        for j in range(np.shape(gate_X_grid)[1]):
            
            slope_1_epsilon0 = slope_1_base
            slope_1 = slope_1_epsilon0-slope_gate_var_1X*gate_X_grid[i,j,:,:]
            slope_2_epsilon0 = min(slope_2_base+slope_occ_var_22*(j-1)*(j>0),slope_2_max)
            slope_2 = slope_2_epsilon0-slope_gate_var_2X*gate_X_grid[i,j,:,:]
            slope_2 = np.maximum(slope_1,slope_2)
            
            slope_interdot = slope_interdot_base[i,j]+slope_occ_var_i2*(j-1)*(j>0)
            
            inter_ratio = (1+slope_interdot*(-1/slope_1))/(1+slope_interdot*(-1/slope_2))
            
            detuning_1[i,j,:,:] = gate_X_grid[i,j,:,:]-1/(slope_1)*gate_Y_grid[i,j,:,:]
            detuning_2[i,j,:,:] = gate_X_grid[i,j,:,:]-1/(slope_2)*gate_Y_grid[i,j,:,:]
            detuning_2[i,j,:,:] *= inter_ratio
            
            detuning_1[i,j,:,:] /= mV_per_EC1
            detuning_2[i,j,:,:] /= mV_per_EC1
            
            EC2[i,j] = (1+slope_interdot*(-1/slope_1_epsilon0))/(1+slope_interdot*(-1/slope_2_epsilon0))*mV_per_EC2/mV_per_EC1

    
    detuning_1 *= -1
    detuning_2 *= -1
    

    # detuning_1 = gate_X_grid-1/(slope_1)*gate_Y_grid
    
    # detuning_2 = inter_ratio*(gate_X_grid-1/(slope_2)*gate_Y_grid)
    # detuning_2 = -detuning_2
    
    return detuning_1,detuning_2,EC2

def extract_charging_energy_2(mu_1,mu_2):
    max_occ2 = np.shape(mu_2)[1]
    
    for i in range(1,max_occ2-1):
        
        energy_of_transition_1 = mu_1[2,i-1,:,:]
        energy_of_transition_2 = mu_2[1,i,:,:]
        
        energy_of_transition_1_other = mu_1[1,i,:,:]
        # transition_1_near_zero = np.abs(energy_of_transition_2)<0.1
        # transition_2_near_zero = np.abs(energy_of_transition_2)<0.1
        # both_near_zero = np.abs(energy_of_transition_1)+np.abs(energy_of_transition_2)<0.01
        both_near_zero_idx = np.argmin(np.abs(energy_of_transition_1)+np.abs(energy_of_transition_2))
        both_other_near_zero_idx = np.argmin(np.abs(energy_of_transition_1_other)+np.abs(energy_of_transition_2))
        
        next_energy_of_transition_2 = mu_2[1,i+1,:,:]
        EC_2 = np.ndarray.flatten(next_energy_of_transition_2)[both_near_zero_idx]
        EC_2_other = np.ndarray.flatten(next_energy_of_transition_2)[both_other_near_zero_idx]
        print(f"EC_2 for transition {i}-->{i+1} between {EC_2} and {EC_2_other}")
    

def make_sim():
    N_gateX, N_gateY = 800,1000
    min_x,max_x = -200,150
    min_y,max_y = -200,200
    
    target = 'BL_new'
    if target == 'BL':
        n_1, n_2 = 12,7
        detuning_1, detuning_2,EC2 = get_detunings_BL(n_1=n_1,n_2=n_2,N_gateX=N_gateX,N_gateY=N_gateY,min_x=min_x,max_x=max_x,min_y=min_y,max_y=max_y)
        mu_1, mu_2 = make_energy_levels_BL(EC2=EC2,n_1=n_1,n_2=n_2,N_gateX=N_gateX,N_gateY=N_gateY)
    elif target == 'SSL':
        n_1, n_2 = 12,5
        detuning_1, detuning_2,EC2 = get_detunings_SSL(n_1=n_1,n_2=n_2,N_gateX=N_gateX,N_gateY=N_gateY,min_x=min_x,max_x=max_x,min_y=min_y,max_y=max_y)
        mu_1, mu_2 = make_energy_levels_SSL(EC2=EC2,n_1=n_1,n_2=n_2,N_gateX=N_gateX,N_gateY=N_gateY)
    elif target == 'BL_new':
        n_1, n_2 = 13,6
        detuning_1, detuning_2,EC2 = get_detunings_BL_new(n_1=n_1,n_2=n_2,N_gateX=N_gateX,N_gateY=N_gateY,min_x=min_x,max_x=max_x,min_y=min_y,max_y=max_y)
        mu_1, mu_2 = make_energy_levels_BL_new(EC2=EC2,n_1=n_1,n_2=n_2,N_gateX=N_gateX,N_gateY=N_gateY)
    elif target == 'test':
        min_x,max_x = -150,100
        min_y,max_y = -200,100
        n_1, n_2 = 10,7
        detuning_1, detuning_2,EC2 = get_detunings_test(n_1=n_1,n_2=n_2,N_gateX=N_gateX,N_gateY=N_gateY,min_x=min_x,max_x=max_x,min_y=min_y,max_y=max_y)
        mu_1, mu_2 = make_energy_levels_test(EC2=EC2,n_1=n_1,n_2=n_2,N_gateX=N_gateX,N_gateY=N_gateY)

    
    
    
    mu_1 -= detuning_1
    mu_2 -= detuning_2
    
    occupation = np.empty((2,np.shape(mu_1)[2],np.shape(mu_1)[3]))

    for i in range(np.shape(mu_1)[2]):
        for j in range(np.shape(mu_1)[3]):
            n_1 = 0
            n_2 = 0
            
            add_1 = mu_1[n_1+1,n_2,i,j]
            add_2 = mu_2[n_1,n_2+1,i,j]
            
            while (add_1<0 or add_2<0):

                if add_1<add_2:
                    n_1+=1
                else:
                    n_2+=1
                add_1 = mu_1[n_1+1,n_2,i,j]
                add_2 = mu_2[n_1,n_2+1,i,j]
                
            occupation[:,i,j] = np.array([2*n_1,n_2])
    
    
    extract_charging_energy_2(mu_1,mu_2)
   
    diff_X = occupation-np.roll(occupation,shift=1,axis=1)
    diff_Y = occupation-np.roll(occupation,shift=1,axis=2)
    

    # print(occupation[:,180,434])
    # print(mu_1[:7,:2,180,434])
    # print(mu_2[:7,:2,180,434])    
    
    diff_XY = np.array([np.abs(diff_X),np.abs(diff_Y)])
    diff = np.max(diff_XY,axis=0)
    total_diff = np.squeeze(np.sum(diff,axis=0))    
    total_diff = total_diff.T
    
    x_gates = np.linspace(min_x,max_x,N_gateX)
    y_gates = np.linspace(min_y,max_y,N_gateY)
    
    x_gates = x_gates[2:]
    y_gates = y_gates[2:]
    total_diff = total_diff[2:,:]
    total_diff = total_diff[:,2:]
    
    # x_gates = np.arange(N_gateX)
    # y_gates = np.arange(N_gateY)
    
    return x_gates,y_gates,total_diff
    
    
def main():
    
    x_gates,y_gates,total_diff = make_sim()
    
    total_diff *= total_diff<5
    # total_diff /= np.max(total_diff)
    # total_diff = np.ceil(total_diff)
    
    plt.figure()
    plt.pcolormesh(x_gates,y_gates,total_diff)
    plt.show()
    

if __name__ == "__main__":
    main()