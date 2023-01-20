import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# import scipy.ndimage
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider, Button, TextBox, RadioButtons
from manual_fitter import manual_fitter

import os
import sys

import time

import core_tools as ct
from core_tools.data.ds.data_set import load_by_uuid

from pathlib import Path

import charge_stability_energyLevels

# if __name__ == '__main__' and __package__ is None:
#     file = Path(__file__).resolve()
#     parent, top = file.parent, file.parents[1]

#     sys.path.append(str(top))
#     try:
#         sys.path.remove(str(parent))
#     except ValueError: # Already removed
#         pass

#     from Simulation import charge_stability_energyLevels
#     __package__ = 'Simulation.charge_stability_energyLevels'


def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

def print_info(ds):
    params = ds.snapshot['station']['instruments']['gates']['parameters']
    sweep_params = ds.m1.x.label.split()
    print(f'\n\nid = {ds.exp_id}')
    print(f'uuid = {ds.exp_uuid}')
    for key in params:
        # print(type(key))
        ds.m1.x.label.split()
        if params[key]['value']!=0 and key!='IDN' and (key not in sweep_params):
            print(f'{key}\t = {round(params[key]["value"])} {params[key]["unit"]}',)
            
#%% Start Browser

# setup logging open database
path = r'C:\Users\aivlev\Documents\Projects\DQW\Data Analysis\ct_config_office.yaml'
ct.configure(path)

# start in separate processes
ct.launch_databrowser()



#%% Load Data Set

# Nicest regime
UUID = 1667576935670283691

#Coulomb Diamonds
# UUID = 1667642554466283691 #BL=-180
# ds = load_by_uuid(1667758099394283691)
# UUID = 1667647198101283691 #BL =-150
# ds = load_by_uuid(1667652271560283691)# BL =-250


#other regime
# UUID = 1666715736964283691 # BL
# UUID = 1666728792682283691 # SSL
# UUID = 1666741845596283691 # BLU
# UUID = 1666755002626283691 # BLD

# ds = load_by_uuid(1666941265380283691) # BL vs SSL

# Other regime still. Interesting for Coulomb Diamonds analysis
# UUID = 1667738841895283691 # BL vs SL
# UUID = 1667747107927283691 # BL = -140 mV
# UUID = 1667758099394283691 # BL = -280 mV

ds = load_by_uuid(UUID)


print_info(ds)

#%%
current_dir = os.path.dirname(os.path.realpath(__file__))
target_dir = current_dir+f"\\hexplotting\\UUID {UUID}"
os.chdir(target_dir)

ds_fitter = manual_fitter.HexPlotter(ds,charge_stability=True,differential=False,cubic=False) 
fig, ax = ds_fitter.make_plot(vmax=None,lognorm=False)
### charge_stability = True indicates that we are looking at the charge diagram, as opposed to a 

# ds_fitter = manual_fitter.MaxPlotter(ds,charge_stability=True,differential=False,cubic=False) 
# ds_fitter.make_plot(vmax=None)

#%%
# load_file = 'Hexgraph_2022_11_30_11_30.pickle'
load_file = 'working_fit.pickle'
# load_file = 'good_region.pickle'
ds_fitter.load_graph(load_file)
#%%
fig, ax = ds_fitter.plot_bare(vmax=None,lognorm=True)
ax.set_xlim(-1.37,-1.18) #BL_new
ax.set_ylim(-0.3,-0.03) #BL_new
# ax.set_xlim(-1.35,-1.1) #BL, SSL
# ax.set_ylim(-0.2,0.1) #BL, SSL
ax.set_ylabel("BL (V)") #BL, SSL


#%%
viridis = mpl.cm.get_cmap('viridis', 256)
newcolors = viridis(np.linspace(0, 1, 256))
red = np.array([232/256, 1/256, 1/256, 1])
orange = np.array([232/256, 131/256, 131/256, 1])
blue = np.array([112/256, 185/256, 185/256, 1])
newcolors[5:90, :] = orange
newcolors[90:250, :] = blue
newcolors[250:256, :] = red
newcmp = mpl.colors.ListedColormap(newcolors)

x_values_sim, y_values_sim, honeycomb = charge_stability_energyLevels.make_sim()
# x_values_sim = x_values_sim/1000-1.3055
# y_values_sim = y_values_sim/1000-0.13

offset_x = -1.215 # BL_new
offset_y = -0.16 # BL_new
# offset_x = -1.166 # BL
# offset_y = -0.136 # BL
# offset_x = -1.182 # SSL
# offset_y = -0.067 # SSL
x_values_sim = x_values_sim/1000+offset_x
y_values_sim = y_values_sim/1000+offset_y


# honeycomb = np.ceil(honeycomb/np.max(honeycomb)) 
# honeycomb = np.roll(honeycomb,1,axis=0)+honeycomb
# honeycomb = np.roll(honeycomb,1,axis=1)+honeycomb

overlay = True
if overlay:
    Dashed = True
    if Dashed:
        compare_val = 0.3
    else:
        compare_val = -1
    honeycomb_mask = np.ones(np.shape(honeycomb)) #to create dashed pattern of the simulation 
    honeycomb_mask = ((np.cumsum(honeycomb_mask,axis=0)%10)/10)>compare_val#>0.3
    honeycomb *= honeycomb_mask
    honeycomb_alpha = 0.6*(honeycomb > 0.1)
else:
    honeycomb_alpha = 1
    fig, ax = plt.subplots()


# ax.pcolormesh(x_values_sim, y_values_sim, honeycomb, shading='auto',alpha=honeycomb_alpha)
ax.pcolormesh(x_values_sim, y_values_sim, honeycomb, shading='auto',alpha=honeycomb_alpha,cmap=newcmp)

honeycomb2 = np.roll(honeycomb,1,axis=0)*honeycomb_mask
honeycomb_alpha2 = 0.6*(honeycomb2 > 0.1)
ax.pcolormesh(x_values_sim, y_values_sim, honeycomb2, shading='auto',alpha=honeycomb_alpha2,cmap=newcmp)

honeycomb3 = np.roll(honeycomb,1,axis=1)*honeycomb_mask
honeycomb_alpha3 = 0.6*(honeycomb3 > 0.1)
ax.pcolormesh(x_values_sim, y_values_sim, honeycomb3, shading='auto',alpha=honeycomb_alpha3,cmap=newcmp)

honeycomb4 = np.roll(np.roll(honeycomb,1,axis=0),1,axis=1)*honeycomb_mask
honeycomb_alpha4 = 0.6*(honeycomb3 > 0.1)
ax.pcolormesh(x_values_sim, y_values_sim, honeycomb4, shading='auto',alpha=honeycomb_alpha4,cmap=newcmp)
plt.xlabel("P1 (mV)")
plt.ylabel("P2 (mV)")
_ = plt.title('Charge stability diagram')

ax.set_xlim(-1.37,-1.18) #BL_new
ax.set_ylim(-0.3,-0.03) #BL_new
# ax.set_xlim(-1.35,-1.1) #BL, SSL
# ax.set_ylim(-0.2,0.1) #BL, SSL


plt.show()

#%%
ds_fitter.plot_difference()