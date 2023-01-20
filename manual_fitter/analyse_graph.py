# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:21:28 2022

@author: aivlev
"""

import numpy as np
import pickle
import itertools
import os
from matplotlib import pyplot as plt
import pyvista

def vertID_to_posind(vertID,vertex_collection):
    return vertex_collection['vertex_list'].index(vertID)

def load_graph(UUID,file):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    target_dir = os.path.abspath(os.path.join(current_dir, '..','..','..', "Projects","DQW","Data Analysis"))
    target_dir = target_dir+f"\\hexplotting\\UUID {UUID}"
    
    os.chdir(target_dir)
    
    f = open(file, 'rb')
    graph_dict = pickle.load(f)  
    
    vertex_collection = graph_dict['vertex_collection']
    edge_collection = graph_dict['edge_collection']
    # x_points = vertex_collection['xy_lists'][0]
    # y_points = vertex_collection['xy_lists'][1]
    
    plt.close()
    
    for key in list(edge_collection.keys()):
        if np.size(edge_collection[key]['vertices_graph'])==0:
            edge_collection.pop(key)
    
    return vertex_collection, edge_collection

def edge_slope_pairs(vertex_pair,vertex_collection):
    pair_posind = []
    pair_x = []
    pair_y = []
    for i in range(2):
        pair_posind.append(vertID_to_posind(vertex_pair[i],vertex_collection))
        pair_x.append(vertex_collection['xy_lists'][0][pair_posind[i]])
        pair_y.append(vertex_collection['xy_lists'][1][pair_posind[i]])
    return pair_x,pair_y
    
def edge_slope_centers(vertex_pair,vertex_collection):
    pair_x, pair_y = edge_slope_pairs(vertex_pair,vertex_collection)
    centers_xy = [(pair_x[0]+pair_x[1])/2,(pair_y[0]+pair_y[1])/2]
    slope = (pair_y[1]-pair_y[0])/(pair_x[1]-pair_x[0])
    return slope,centers_xy

def edge_slope_crossing(vertex_pair,vertex_collection,cross_value,axis_value):
    pair_x, pair_y = edge_slope_pairs(vertex_pair,vertex_collection)
    pairs_oriented = [pair_x,pair_y]
    pairs_oriented = [pairs_oriented[(axis_value+1)%2],pairs_oriented[axis_value]]
    if sum(np.array(pairs_oriented[1]) < cross_value)!=1:
        return None
    slope = (pairs_oriented[1][1]-pairs_oriented[1][0])/(pairs_oriented[0][1]-pairs_oriented[0][0])
    cross_at_pos = (cross_value-pairs_oriented[1][0])/slope+pairs_oriented[0][0]
    return cross_at_pos
        

def slope_vs_gate(vertex_collection,edge_collection):           
    slopes_collection = {}
        
    for key in edge_collection.keys():
        if np.size(edge_collection[key]['vertices_graph'])>0:
            slope_list = []
            centers_xy_list = []
            for i in range(np.shape(edge_collection[key]['vertices_graph'])[0]):
                slope, centers_xy = edge_slope_centers(edge_collection[key]['vertices_graph'][i],vertex_collection)
                slope_list.append(slope)
                centers_xy_list.append(centers_xy)
            slope_list = np.array(slope_list)
            centers_xy_list = np.array(centers_xy_list)

            points = np.array([centers_xy_list[:,0],centers_xy_list[:,1],slope_list])
            points = points.T
            
            plane, center, normal = pyvista.fit_plane_to_points(points, return_meta=True)
            normal = normal/normal[2]
            
            slopes_collection[key] = {'x_y_slopes':points,'fit':{'center':center,'normal':normal}}
    return slopes_collection
    
def plot_slopes_vs_gates(slopes_collection,xy=None,plot_fit=False,plot_3D=False,plot_maxRes=True,ax_x=None,ax_y=None,ax_maxRes=None,ax_all=None):
    color_dict = {'low':'orange','mid':'cyan','high':'red','unknown':'green'}
    
    if ax_x is None:
        plt.figure()
        ax_x = plt.axes()
    if ax_y is None:
        plt.figure()
        ax_y = plt.axes()
    if ax_maxRes is None and plot_maxRes:
        plt.figure()
        ax_maxRes = plt.axes()
    
    if ax_all is None and plot_3D:
        plt.figure()
        ax_all = plt.axes(projection='3d')
    
    for key in slopes_collection.keys():
        centers_xy_list = slopes_collection[key]['x_y_slopes'][:,0:2]
        slope_list = slopes_collection[key]['x_y_slopes'][:,2]
        normal = slopes_collection[key]['fit']['normal']
        center = slopes_collection[key]['fit']['center']
        d = -center.dot(normal)
        
        x_line = np.linspace(min(centers_xy_list[:,0]),max(centers_xy_list[:,0]))
        y_line = np.linspace(min(centers_xy_list[:,1]),max(centers_xy_list[:,1]))
        
        if xy!= 1:
            ax_x.scatter(centers_xy_list[:,0],slope_list,c=color_dict[key]) 
            ax_x.set_xlabel('x-gate')
            ax_x.set_ylabel('Slope')
            
            if plot_fit:
                z_x = (-normal[0]*x_line - normal[1]*center[1] - d) * 1. /normal[2]
                ax_x.plot(x_line,z_x,color=color_dict[key])
        if xy!=0:
            ax_y.scatter(centers_xy_list[:,1],slope_list,c=color_dict[key]) 
            # ax_y.set_xlabel('y-gate')
            # ax_y.set_ylabel('Slope')
            ax_y.set_xlabel('BL at mid-point of transition (V)')
            ax_y.set_ylabel('Slope (BL/SL) of transition')
            
            if plot_fit and key != 'high':
                z_y = (-normal[1]*y_line - normal[0]*center[0] - d) * 1. /normal[2]
                ax_y.plot(y_line,z_y,color=color_dict[key])

        
        if plot_maxRes:    
        
            xx, yy = np.meshgrid(x_line,y_line)
            
            # calculate corresponding z
            z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
            
            if plot_3D:
                ax_all.scatter3D(centers_xy_list[:,0], centers_xy_list[:,1], slope_list, color = color_dict[key])
                ax_all.plot_surface(xx,yy,z,alpha=0.2,color = color_dict[key])
            
            normal_0 = slopes_collection['low']['fit']['normal']        
            v_best_0 = [normal_0[0],normal_0[1]]/(normal_0[0]**2+normal_0[1]**2)**0.5
            
            normal_1 = slopes_collection['mid']['fit']['normal']        
            v_best_1 = [normal_1[0],normal_1[1]]/(normal_1[0]**2+normal_1[1]**2)**0.5
            
            v_best = (v_best_0+v_best_1)/2
            
            for key in slopes_collection.keys():
                centers_xy_list = slopes_collection[key]['x_y_slopes'][:,0:2]
                slope_list = slopes_collection[key]['x_y_slopes'][:,2]
                center = slopes_collection[key]['fit']['center']
                best_projection = (centers_xy_list-center[0:2]).dot(v_best)
                ax_maxRes.scatter(best_projection,slope_list,color = color_dict[key])
                ax_maxRes.set_ylabel('Slope')
                ax_maxRes.set_xlabel(f'Voltage along direction ({v_best[0]:3.2f},{v_best[1]:3.2f})(SL,BL)')

def mean_across_gates(sl_vs_gate_UUID,graph_file):
    mean_vs_gate = []
    gate_names = []
    
    for gate in sl_vs_gate_UUID:
        gate_names.append(gate)
        UUID = sl_vs_gate_UUID[gate]
        vertex_collection, edge_collection = load_graph(UUID,graph_file)
        slopes_collection = slope_vs_gate(vertex_collection,edge_collection)
        
        mean_slopes = []
        for key in slopes_collection.keys():
            slopes = slopes_collection[key]['x_y_slopes'][:,2]
            mean = np.mean(slopes)
            std = np.std(slopes)
            mean_slopes.append([mean,std])
        mean_vs_gate.append(mean_slopes)
    mean_vs_gate = np.array(mean_vs_gate)
    print(np.shape(mean_vs_gate))
    
    plt.figure()
    plt.title("Mean slopes for different Gate-SL CS-diagrams")
    color_list = ['orange','cyan','red','green']
    
    plt_xaxis = np.arange(np.shape(mean_vs_gate)[0])
    for i in range(np.shape(mean_vs_gate)[1]):
        plt.errorbar(plt_xaxis,mean_vs_gate[:,i,0],mean_vs_gate[:,i,1],fmt='-o',capsize=5,color=color_list[i])
    plt.xticks(plt_xaxis,gate_names)
    plt.ylabel('Mean Slope')
    
    ### Now coupling instead of slopes
    plt.figure("Mean Relative Couplings")
    plt.title("Mean relative coupling of [GATE]-P, for each dot")
    
    plt_xaxis = np.arange(np.shape(mean_vs_gate)[0])
    for i in range(np.shape(mean_vs_gate)[1]-1):
        std = mean_vs_gate[:,i,1]/(mean_vs_gate[:,i,0])**2
        plt.errorbar(plt_xaxis,-1/mean_vs_gate[:,i,0],std,fmt='-o',capsize=5,color=color_list[i])
    gate_names =['PB','$S_{SHT}$','$B_S$','$B_D$']
    plt.xticks(plt_xaxis,gate_names)
    plt.ylabel('Relative Coupling')
    plt.ylim((0,1))
    

def value_crossing_slopes(vertex_collection,edge_collection,value=0,xy='y'):
    crossing_slopes_collection = {'cross_axis' : xy}
    
    if xy == 'x':
        axis_value = 0
    elif xy == 'y':
        axis_value = 1
    else:
        print('WARNING: XY INPUT CHOICE UNEXPECTED')
    
    for key in edge_collection.keys():
        if np.size(edge_collection[key]['vertices_graph'])>0:
            slope_list = []
            crossing_pos_list = []
            for i in range(np.shape(edge_collection[key]['vertices_graph'])[0]):
                crossing_pos = edge_slope_crossing(edge_collection[key]['vertices_graph'][i],vertex_collection,value,axis_value)
                if not (crossing_pos is None):
                    slope, _ = edge_slope_centers(edge_collection[key]['vertices_graph'][i],vertex_collection)
                    crossing_pos_list.append(crossing_pos)
                    slope_list.append(slope)
            if len(crossing_pos_list) > 1:
                temp = np.array([crossing_pos_list,slope_list])
                temp = temp[:,temp[0,:].argsort()]
                crossing_pos_list = temp[0,:]
                slope_list = temp[1,:]
            crossing_slopes_collection[key] = {'crossing_pos_list' : crossing_pos_list, 
                    'slope_list' : slope_list
                    }
    return crossing_slopes_collection

def correspondance_across_gates(sl_vs_gate_UUID,graph_file,value_dict={'BL':0},xy='y'):
    color_dict = {'low':'orange','mid':'cyan','high':'red','unknown':'green'}
    gate_names = []
    crossing_master_dict = {}
    epsilon = 0.003 # V
    
    plt.figure()
    plt.title("Slopes of the same transition across different [GATE]-SL CS-diagrams")
    
    
    for gate in sl_vs_gate_UUID:
        gate_names.append(gate)
        UUID = sl_vs_gate_UUID[gate]
        vertex_collection, edge_collection = load_graph(UUID,graph_file)
        
        value = value_dict[gate]
        
        crossing_slopes_collection = value_crossing_slopes(vertex_collection,edge_collection,value=value,xy=xy)
        crossing_gate_list = []
        for key in edge_collection.keys():
            if not (key in crossing_master_dict.keys()):
                crossing_master_dict[key] = {}
            crossing_master_dict[key][gate] = [crossing_slopes_collection[key]['crossing_pos_list'],crossing_slopes_collection[key]['slope_list']]

    for key in crossing_master_dict.keys():
        gate_list = list(crossing_master_dict[key].keys())
        slopes_per_gate_mask = []
        slopes_per_gate = []
        crossing_slopes_list = np.empty((2,0))
        for index, gate in enumerate(gate_list):
            temp = np.array(crossing_master_dict[key][gate])
            slopes_per_gate_mask += (np.shape(temp)[1]*[index])
            slopes_per_gate.append(np.shape(temp)[1])
            crossing_slopes_list = np.append(crossing_slopes_list,temp,axis=1)
            # pass
        crossing_slopes_list = np.append(crossing_slopes_list,[slopes_per_gate_mask],axis=0)

        
        corresponding_list = []
        while np.size(crossing_slopes_list)>0:
            crossings = crossing_slopes_list[0,:]
            close_indices = (np.abs(crossings-crossings[0])<epsilon)
            corresponding_set = crossing_slopes_list[:,close_indices]
            plt.plot(corresponding_set[2,:],corresponding_set[1,:],'o-',color=color_dict[key])
            crossing_slopes_list = crossing_slopes_list[:,~close_indices]
        plt.xticks(range(len(gate_list)),gate_list)
        plt.ylabel('Corresponding Slope')
        
        
    ### Now coupling instead of slopes
    plt.figure()
    plt.title("Relative coupling of [GATE]-SL, for each dot")
    
    for gate in sl_vs_gate_UUID:
        gate_names.append(gate)
        UUID = sl_vs_gate_UUID[gate]
        vertex_collection, edge_collection = load_graph(UUID,graph_file)
        
        value = value_dict[gate]
        
        crossing_slopes_collection = value_crossing_slopes(vertex_collection,edge_collection,value=value,xy=xy)
        crossing_gate_list = []
        for key in edge_collection.keys():
            if not (key in crossing_master_dict.keys()):
                crossing_master_dict[key] = {}
            crossing_master_dict[key][gate] = [crossing_slopes_collection[key]['crossing_pos_list'],crossing_slopes_collection[key]['slope_list']]

    for key in crossing_master_dict.keys():
        if key != 'high':
            gate_list = list(crossing_master_dict[key].keys())
            slopes_per_gate_mask = []
            slopes_per_gate = []
            crossing_slopes_list = np.empty((2,0))
            for index, gate in enumerate(gate_list):
                temp = np.array(crossing_master_dict[key][gate])
                slopes_per_gate_mask += (np.shape(temp)[1]*[index])
                slopes_per_gate.append(np.shape(temp)[1])
                crossing_slopes_list = np.append(crossing_slopes_list,temp,axis=1)
                # pass
            crossing_slopes_list = np.append(crossing_slopes_list,[slopes_per_gate_mask],axis=0)
    
            
            corresponding_list = []
            while np.size(crossing_slopes_list)>0:
                crossings = crossing_slopes_list[0,:]
                close_indices = (np.abs(crossings-crossings[0])<epsilon)
                corresponding_set = crossing_slopes_list[:,close_indices]
                plt.plot(corresponding_set[2,:],-1/corresponding_set[1,:],'o-',color=color_dict[key])
                crossing_slopes_list = crossing_slopes_list[:,~close_indices]
            plt.xticks(range(len(gate_list)),gate_list)
            plt.ylim((0,1))
            plt.ylabel('Relative Coupling')
            

def slope_across_vertices(vertex_collection,edge_collection):
    color_dict = {'low':'orange','mid':'cyan','high':'red','unknown':'green'}
    
    vertex_list = vertex_collection['vertex_list']
    vertex_list = np.array(vertex_list)
    
    plt.figure()
    plt.title('Slopes connected to different vertices')
    plt.xlabel('(Arbitrary) vertex ID')
    plt.ylabel('Slope')
    
    for key in edge_collection.keys():
        present_vertID = []
        corresponding_slope = []
        if len(edge_collection[key]['vertices_graph']) > 0:
            for vertexID in vertex_list:
                condition = np.any(edge_collection[key]['vertices_graph'] == np.array(vertexID),axis=1)
                connecting_pairs = list(itertools.compress(edge_collection[key]['vertices_graph'],condition))
                if len(connecting_pairs) > 1:
                    print('Same vertex multiple identical slopes!')
                elif len(connecting_pairs) == 1:
                    present_vertID.append(vertexID)
                    slope,_ = edge_slope_centers(connecting_pairs[0],vertex_collection)
                    corresponding_slope.append(slope)
            plt.plot(present_vertID,corresponding_slope,'o',color = color_dict[key])
    
    plt.show()
    
def determine_hexagons(vertex_collection,edge_collection):
    vertex_list = vertex_collection['vertex_list']
    visited_vertices = [vertex_list[0]]
    mapped_vertices = 100*np.ones((3,len(vertex_list)))    
    
    low_transition_list = np.array(edge_collection['low']['vertices_graph'])
    mid_transition_list = np.array(edge_collection['mid']['vertices_graph'])
    high_transition_list = np.array(edge_collection['high']['vertices_graph'])
    transition_list = [low_transition_list,mid_transition_list,high_transition_list]
       
    low_check = np.any(low_transition_list == np.array([visited_vertices[0]]),axis=1)
    mid_check = np.any(mid_transition_list == np.array([visited_vertices[0]]),axis=1)
    high_check = np.any(high_transition_list == np.array([visited_vertices[0]]),axis=1)
    
    if np.any(low_check):
        relevant_transition_list = low_transition_list
        relevant_check = low_check
        additional_factor = 1
        
    elif np.any(high_check):
        relevant_transition_list = high_transition_list
        relevant_check = high_check
        additional_factor = 1
        
    elif np.any(mid_check):
        relevant_transition_list = mid_transition_list
        relevant_check = mid_check
        additional_factor = -1
    
    pair = relevant_transition_list[relevant_check]
    other_vertex = pair[pair!=visited_vertices[0]]
    vertex_y = vertex_collection['xy_lists'][1][np.where(vertex_list == np.array(visited_vertices[0]))[0][0]]
    other_y = vertex_collection['xy_lists'][1][np.where(vertex_list == np.array(other_vertex))[0][0]]
    hexagon_indicator = additional_factor*np.sign(other_y-vertex_y)
    tmp = hexagon_indicator
    hexagon_indicator = 1-(hexagon_indicator+1)/2
    hexagon_indicator /= 2
    mapped_vertices[:,np.where(vertex_list == np.array(visited_vertices[0]))[0][0]] = [hexagon_indicator,hexagon_indicator,tmp]
    
    low_transition_remaining = low_transition_list
    mid_transition_remaining = mid_transition_list
    high_transition_remaining = high_transition_list
    
    transitions_remaining = [low_transition_remaining,mid_transition_remaining,high_transition_remaining]
    
    mapped_low = 100*np.ones((2,len(low_transition_list)))
    mapped_mid = 100*np.ones((2,len(mid_transition_list)))
    mapped_high = 100*np.ones((2,len(high_transition_list)))
    
    mapped_transitions = [mapped_low,mapped_mid,mapped_high]
    
    transitions_remaining_count = len(low_transition_remaining)+len(mid_transition_remaining)+len(high_transition_remaining)

    while transitions_remaining_count>0:
        
        for i in range(len(visited_vertices)):
            low_check = np.any(transitions_remaining[0] == np.array([visited_vertices[i]]),axis=1)
            mid_check = np.any(transitions_remaining[1] == np.array([visited_vertices[i]]),axis=1)
            high_check = np.any(transitions_remaining[2] == np.array([visited_vertices[i]]),axis=1)
            if np.any(low_check):
                relevant_transition_type = 0
                relevant_check = low_check
                change_factor = np.array([0.5,-0.5])
                something_found = True
                break
            elif np.any(mid_check):
                relevant_transition_type = 1
                relevant_check = mid_check
                change_factor = np.array([-0.5,0.5])
                something_found = True
                break
            elif np.any(high_check):
                relevant_transition_type = 2
                relevant_check = high_check
                change_factor = np.array([-0.5,-0.5])
                something_found = True
                break
            else:
                something_found = False
        
        if not something_found:
            print("There are unconnected transitions remaining!!")
            plt.figure()
            for i in range(len(visited_vertices)):
                vertex_idx = np.where(vertex_list == np.array(visited_vertices[i]))[0][0]
                plt.plot(vertex_collection['xy_lists'][0][vertex_idx],vertex_collection['xy_lists'][1][vertex_idx],'.')
            plt.show()
            print(visited_vertices)
            print(transitions_remaining)
            
        remaining_pair_idx = np.arange(len(relevant_check))[relevant_check][0]
        pair = transitions_remaining[relevant_transition_type][remaining_pair_idx]
        pair_idx = list(map(list,list(transition_list[relevant_transition_type]))).index(list(pair))
        
        other_vertex = pair[pair!=visited_vertices[i]][0]
        
        visited_vertices.append(other_vertex)
        
        vertex_idx = np.where(vertex_list == np.array(visited_vertices[i]))[0][0]
        other_vertex_idx = np.where(vertex_list == np.array(other_vertex))[0][0]
        old_vertex_hexagon = mapped_vertices[:,vertex_idx]
        mapped_vertices[0:2,other_vertex_idx] = old_vertex_hexagon[0:2]+old_vertex_hexagon[2]*change_factor
        mapped_vertices[2,other_vertex_idx] = -old_vertex_hexagon[2]
        
        if old_vertex_hexagon[2]==1:
            mapped_transitions[relevant_transition_type][:,pair_idx] = old_vertex_hexagon[0:2]
        elif old_vertex_hexagon[2]==-1:
            mapped_transitions[relevant_transition_type][:,pair_idx] = mapped_vertices[0:2,other_vertex_idx]
        else:
            print('SOMETHING IS WRONG')
        
        transitions_remaining[relevant_transition_type] = np.delete(transitions_remaining[relevant_transition_type],remaining_pair_idx,axis=0)
        transitions_remaining_count = len(transitions_remaining[0])+len(transitions_remaining[1])+len(transitions_remaining[2])
            
    
    color_list = ['orange','cyan','red']
    
    
    fig_dot1, ax_dot1 = plt.subplots(ncols=1)
    fig_dot2, ax_dot2 = plt.subplots(ncols=1)
    
    ax_dot1.set_title('Relative Occupation Dot 1 vs Slope')
    ax_dot2.set_title('Relative Occupation Dot 2 vs Slope')
    
    ax_dot1.set_xlabel('Relative Occupation Dot 1')
    ax_dot2.set_xlabel('Relative Occupation Dot 2')
    
    ax_dot1.set_ylabel('Slope')
    ax_dot2.set_ylabel('Slope')
    
    
    for transition_type_idx in range(len(transition_list)):
        for vertex_pair_idx in range(len(transition_list[transition_type_idx])):
            slope,_ = edge_slope_centers(transition_list[transition_type_idx][vertex_pair_idx],vertex_collection)
            
            relevant_hexagon = mapped_transitions[transition_type_idx][:,vertex_pair_idx]
            ax_dot1.plot(relevant_hexagon[0],slope,'.',color=color_list[transition_type_idx])
            ax_dot2.plot(relevant_hexagon[1],slope,'.',color=color_list[transition_type_idx])
    plt.show()    
    
    keys = ['low','mid']
    colors = ['orange','cyan']
    plt.figure()
    for transition_type in range(2):
        max_occupation = 0 
        min_occupation = 0
        max_occupation_other = int(max(max_occupation,np.max(mapped_transitions[transition_type][transition_type,:])))
        min_occupation_other = int(min(min_occupation,np.min(mapped_transitions[transition_type][transition_type,:])))
        
        if transition_type == 1:
            slope_fix = -2.2
        else: 
            slope_fix = None
        
        occupation_list = []
        occupation_other_list = []
        EC_list = []
        for occupation_count_other in range(min_occupation_other,max_occupation_other):
            relevant_transitions = mapped_transitions[transition_type][transition_type,:] == np.array(occupation_count_other)
            relevant_transitions = np.arange(len(relevant_transitions))[relevant_transitions]
            for transition in relevant_transitions:
                occupation_count = mapped_transitions[transition_type][1-transition_type,transition]
                next_transition = mapped_transitions[transition_type][1-transition_type,relevant_transitions] == occupation_count+1
                next_transition = relevant_transitions[next_transition]
                
                if len(next_transition)>0:
                    next_transition = next_transition[0]
                    current_pair = edge_collection[keys[transition_type]]['vertices_graph'][transition]
                    next_pair = edge_collection[keys[transition_type]]['vertices_graph'][next_transition]
                    
                    
                    EC_transition = charging_energy_between_transitions(current_pair,next_pair,vertex_collection,slope_fix=slope_fix)
                    occupation_list.append(occupation_count)
                    occupation_other_list.append(occupation_count_other)
                    EC_list.append(EC_transition)
        plt.plot(occupation_list,EC_list,'o',color=colors[transition_type])
        print(f"Transition type: {keys[transition_type]}")
        print(occupation_list)
        print(occupation_other_list)
        print(EC_list)
    plt.title('Ec as function of different level transitions, for both dots at different occupations of other dot')
    plt.ylabel('Ec (mV in x-gate)')
    plt.xlabel('Occupation-level')
    plt.show()
    return mapped_transitions        
            
def dependance_at_occupation(vertex_collection,edge_collection,mapped_transitions):
    dot_occupied = 2
    
    max_occupation = 0 
    min_occupation = 0
    for transition_idx in range(len(mapped_transitions)):
        max_occupation = max(max_occupation,np.max(mapped_transitions[transition_idx][dot_occupied-1,:]))
        min_occupation = min(min_occupation,np.min(mapped_transitions[transition_idx][dot_occupied-1,:]))
    min_occupation = int(min_occupation)
    max_occupation = int(max_occupation)

    
    fig_x, axes_x = plt.subplots(ncols=max_occupation-min_occupation+1)
    fig_y, axes_y = plt.subplots(ncols=max_occupation-min_occupation+1)
    fig_x.suptitle('Gate X vs Slope at different occupation', fontsize=16)
    fig_y.suptitle('Gate Y vs Slope at different occupation', fontsize=16)
    
    occupation_list = range(min_occupation,max_occupation+1)
    for i in range(max_occupation-min_occupation+1):
        occupation_count = occupation_list[i]
        edge_collection_at_occupation = {}
        keys = ['low','mid','high']
        for idx, key in enumerate(keys):
            relevant_transitions = mapped_transitions[idx][dot_occupied-1,:] == np.array(occupation_count)
            tmp = np.array(edge_collection[key]['vertices_graph'])[relevant_transitions]
            edge_collection_at_occupation[key] = {'vertices_graph':list(map(list,list(tmp)))}
        
        slopes_at_occupation = slope_vs_gate(vertex_collection,edge_collection_at_occupation)
        plot_slopes_vs_gates(slopes_at_occupation,xy=None,plot_fit=False,ax_x=axes_x[i],ax_y=axes_y[i],plot_3D=False,plot_maxRes=False)
        axes_x[i].set_title(f"N={occupation_count}")
        axes_x[i].set_ylabel("")
        axes_x[i].set_ylim((-6,-1))
        axes_y[i].set_title(f"N={occupation_count}")
        axes_y[i].set_ylabel("")
        axes_y[i].set_ylim((-6,-1))  
    
def charging_energy_between_transitions(transition_1,transition_2,vertex_collection,slope_fix=None):
    ### Calculating the charging energy at y-gate=0, in mV of the X-gate
    slope_1,center_1 = edge_slope_centers(transition_1,vertex_collection)
    slope_2,center_2 = edge_slope_centers(transition_2,vertex_collection)
    
    if not slope_fix is None:
        slope_1 = slope_fix
        slope_2 = slope_1

    EC = (center_2[0]-center_2[1]/slope_2)-(center_1[0]-center_1[1]/slope_1) #mV
    return EC


def find_end_of_line(vertex,first_transition_list,second_transition_list):
    end_found = False
    end_vertex = vertex
    transition_to_follow = 0
    transitions_combined = [first_transition_list,second_transition_list]
    first_step = True
    visited_pairs = []
    visited_vertex = []
    
    while not(end_found):
        visited_vertex.append(vertex)
        
        first_check = np.any(transitions_combined[0] == np.array([vertex]),axis=1)
        second_check = np.any(transitions_combined[1] == np.array([vertex]),axis=1)
        check_sum = int(np.any(first_check))+int(np.any(second_check))
        if check_sum==1:
            if first_step:
                transition_to_follow = np.array([0,1])[np.array([np.any(first_check),np.any(second_check)])][0]
                
                follow_list = np.array(transitions_combined[transition_to_follow])
                relevant_pair = follow_list[np.any(follow_list == np.array([vertex]),axis=1)]
                vertex = relevant_pair[relevant_pair != vertex][0]
                
                if transition_to_follow == 0:
                    visited_pairs.append(relevant_pair)
                transition_to_follow = 1-transition_to_follow
            else: 
                end_vertex=vertex
                end_found = True
        elif check_sum==0:
            print("NO TRANSITION IS FOUND")
        elif check_sum==2:
            follow_list = np.array(transitions_combined[transition_to_follow])
            relevant_pair = follow_list[np.any(follow_list == np.array([vertex]),axis=1)]
            vertex = relevant_pair[relevant_pair != vertex][0]
            
            if transition_to_follow == 0:
                visited_pairs.append(relevant_pair)
            transition_to_follow = 1-transition_to_follow
        else:
            print("WRONG TRANSITION SUM")
        first_step = False
        
    return visited_vertex,visited_pairs
    
def slope_across_transitions(vertex_collection,vertex_list,reservoir_transition_list,other_list):
    pair_collection = []
    slope_collection = []
    
    while len(vertex_list)>0:
        vertex_to_check = vertex_list[0]
        reservoir_check = np.any(reservoir_transition_list == np.array([vertex_to_check]),axis=1)
        other_check = np.any(other_list == np.array([vertex_to_check]),axis=1)
        if not(np.any(reservoir_check) or np.any(other_check)):
            vertex_list.pop(0)
        else:
            visited_vertex,visited_pairs = find_end_of_line(vertex_to_check,reservoir_transition_list,other_list)
            end_vertex = visited_vertex[-1]
            
            visited_vertex,visited_pairs = find_end_of_line(end_vertex,reservoir_transition_list,other_list)
 
            vertex_list = [element for element in vertex_list if element not in visited_vertex]
            
            slope_list = []
            for i in range(len(visited_pairs)):
                slope, _ = edge_slope_centers(visited_pairs[i][0],vertex_collection)
                slope_list.append(slope)
                
            slope_collection.append(slope_list)
            pair_collection.append(visited_pairs)
            
    return slope_collection, pair_collection
    
def slope_across_transitions_main(vertex_collection,edge_collection,reservoir_transition_name,other_transition_name,interdot_name="high"):
    ### first calculate as function of other transition
    vertex_list = vertex_collection['vertex_list']
    reservoir_transition_list = edge_collection[reservoir_transition_name]['vertices_graph']
    other_transition_list = edge_collection[other_transition_name]['vertices_graph']
    interdot_list = edge_collection[interdot_name]['vertices_graph']
       
    slope_collection, _ = slope_across_transitions(vertex_collection,vertex_list,reservoir_transition_list,interdot_list)               
    
    plt.figure()
    for i in range(len(slope_collection)):
        plt.plot(slope_collection[i])
        
    slope_collection, pair_collection = slope_across_transitions(vertex_collection,vertex_list,reservoir_transition_list,other_transition_list)               
    
    for i in range(len(slope_collection)):
        plt.plot(slope_collection[i])
        
def interdot_at_occupation(vertex_collection,edge_collection,mapped_transitions):
    
    slopes_collection = slope_vs_gate(vertex_collection,edge_collection)
    mean_slopes = []
    for key in slopes_collection.keys():
        slopes = slopes_collection[key]['x_y_slopes'][:,2]
        mean = np.mean(slopes)
        mean_slopes.append(mean)
        
    interdot_num = np.shape(mapped_transitions[2])[1]      
    all_interdots = []
    all_occupation_2 = []
    for i in range(interdot_num):
        occ_1 = mapped_transitions[2][0,i]
        occ_2 = mapped_transitions[2][1,i]
        
        transition_1 = np.array((mapped_transitions[1][0,:] == occ_1)*(mapped_transitions[1][1,:] == occ_2-1))
        transition_2 = np.array((mapped_transitions[0][0,:] == occ_1-1)*(mapped_transitions[0][1,:] == occ_2))
        
        if np.any(transition_1):
            pair_1 = np.array(edge_collection['mid']['vertices_graph'])[transition_1][0]
            slope_1, _ = edge_slope_centers(pair_1,vertex_collection)
            mean_taken_1 = False
            print(slope_1)
        else:
            slope_1 = mean_slopes[1]
            mean_taken_1 = True
        if np.any(transition_2):
            pair_2 = np.array(edge_collection['low']['vertices_graph'])[transition_2][0]
            slope_2, _ = edge_slope_centers(pair_2,vertex_collection)
            mean_taken_2 = False
            print(slope_2)
        else:
            slope_2 = mean_slopes[0]
            print(slope_2)
            input()
            mean_taken_2 = True
        
        pair_interdot = np.array(edge_collection['high']['vertices_graph'])[i]
        slope_interdot, _ = edge_slope_centers(pair_interdot,vertex_collection)
        
        
        interdot = (1+slope_interdot*(-1/slope_1))/(1+slope_interdot*(-1/slope_2))
        all_interdots.append(interdot)
        all_occupation_2.append(occ_2)

    plt.figure()
    plt.plot(all_occupation_2,all_interdots,'.')        
    plt.show()
#%%
#Nicest regime
# UUID = 1667576935670283691

#Coulomb Diamonds
# ds = load_by_uuid(1667642554466283691) #BL=-180
# ds = load_by_uuid(1667758099394283691)
# ds = load_by_uuid(1667647198101283691)# BL =-150
# ds = load_by_uuid(1667652271560283691)# BL =-250


#other regime
# UUID = 1666715736964283691 # BL}
# UUID = 1666728792682283691 # SSL
# UUID = 1666741845596283691 # BLU
# UUID = 1666755002626283691 # BLD

sl_vs_gate_UUID = {'BL':1666715736964283691,
                   'SSL':1666728792682283691,
                   'BLU':1666741845596283691,
                   'BLD':1666755002626283691}

gate_cut_dict = {'BL':0,'SSL':0,'BLU':-0.237,'BLD':-0.178}

    #%%
    
def main():
    plt.close('all')
    
    # UUID = sl_vs_gate_UUID['SSL']
    UUID = 1667576935670283691
       
    graph_file = 'working_fit.pickle'
    # graph_file = 'good_region.pickle'
    
    vertex_collection, edge_collection = load_graph(UUID,graph_file)
    
    if True:
        slopes_collection = slope_vs_gate(vertex_collection,edge_collection)
        plot_slopes_vs_gates(slopes_collection,xy=None,plot_fit=False)
        
        # slope_across_vertices(vertex_collection,edge_collection)
        
        mean_across_gates(sl_vs_gate_UUID,graph_file)
    
        correspondance_across_gates(sl_vs_gate_UUID,graph_file,value_dict=gate_cut_dict,xy='y')
    
    # slope_across_transitions_main(vertex_collection,edge_collection,reservoir_transition_name='low',other_transition_name='mid',interdot_name="high")
    mapped_transitions = determine_hexagons(vertex_collection,edge_collection)
    
    dependance_at_occupation(vertex_collection,edge_collection,mapped_transitions)
    interdot_at_occupation(vertex_collection,edge_collection,mapped_transitions)
    
if __name__ == "__main__":
    main()    
