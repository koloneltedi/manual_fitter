# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 14:50:38 2022

@author: aivlev
"""

import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox, RadioButtons

class PlotObject:
    """
    The class which enables all the plotting. Very ugly use of classes. Very ugly use of object properties. Might improve it with time
    """
    def __init__(self,ds,charge_stability,differential=False,cubic=True):
        """
        Initialised the plotting class. Now it is still in a single class, and a lot of parameters are global. 
        Parameters
        ----------
        charge_stability : boolean
            If plotting a charge_stability diagram as opposed to a Coulomb Diamond (make False in that case)
        differential : boolean ,optional
            Take the differential in the x-axis. Differential is taken away from 0, i.e. it is inverted for negative values of x
        cubic: boolean, optional
            Smooth the heatmap with a cubic curve, before taking a linecut. Otherwise it is nearest neighbour
            .. version added:: 0.1
        """
        
        self.cubic = cubic
        self.parse_dataset(ds,charge_stability,differential)
        self.x0, self.y0 = min(self.x)+0.001, min(self.y)+0.001
        self.x1, self.y1 = max(self.x)-0.001, min(self.y)+0.001

        self.pind = None # Active Point
        self.epsilon = 18 # Max Pixel Distance

        self.x_points = [self.x0,self.x1]
        self.y_points = [self.y0,self.y1]

        self.zi = self._make_linecut(self.x_points[0],self.y_points[0],self.x_points[1],self.y_points[1])

    def parse_dataset(self,ds, charge_stability,differential):
        # Parse Data
    
        self.x = ds.m1.y()/1000 ### FLIP THESE BECAUSE DATABROWSER/BASE IS WEIRD # In V
        self.y = ds.m1.x()/1000 ### FLIP THESE BECAUSE DATABROWSER/BASE IS WEIRD # In V
        self.z = ds.m1.z()
        if charge_stability: #If charge stability
            self.z = np.flip(self.z) ### FLIP THIS BECAUSE DATABROWSER/BASE IS WEIRD. Check if always needed.
            xcut = 2
        else: # Coulomb Diamonds
            if differential:
                dx = self.x[1]-self.x[0]
                dz = (self.z-np.roll(self.z,1,axis=1))*np.tile(np.sign(self.x),(np.size(self.y),1))
                self.z = dz/dx
                xcut = 10 
    
           
        self.x = self.x[xcut:-xcut]
        self.z = self.z[:,xcut:-xcut]
    
        self.lever_x = 1# #eV/V
        self.lever_y = 1# #eV/V
    
        self.lockin_amp = ds._data_set__data_set_raw.snapshot['station']['parameters']['Vsd_ac']['value']
        self.lockin_amp = self.lockin_amp*1e-6 # now in V
    
        self.z = self.z/self.lockin_amp # now in Siemens
        Rk = 12906.4 # Resistance quantum in Ohm
        self.z = self.z*Rk # Now in units of conductance quantum G0 2e^2/h
        


    def _make_linecut(self,x0,y0,x1,y1):
        #Line cut function    
        
        cubic = self.cubic
        x,y,z = self.x, self.y,self.z
        
        #-- Extract the line...
        # Make a line with "num" points...
    
        
        # convert to pixel values
        x_plen, y_plen = np.size(x),np.size(y)
        x_len, y_len = max(x)-min(x), max(y)-min(y)
        
        x0_p, y0_p = (x0-min(x))/x_len*x_plen, (max(y)-y0)/y_len*y_plen # These are in _pixel_ coordinates!!
        x1_p, y1_p = (x1-min(x))/x_len*x_plen, (max(y)-y1)/y_len*y_plen
        
        x0_p, y0_p, x1_p,y1_p = np.array(np.round([x0_p+1,y0_p-1,x1_p-1,y1_p-1]),dtype=int)
        
        if cubic:
            num = 1000
            x_cut, y_cut = np.linspace(x0_p, x1_p, num), np.linspace(y0_p, y1_p, num)
        
            # Extract the values along the line, using cubic interpolation
            zi = scipy.ndimage.map_coordinates(z.T, np.vstack((x_cut,y_cut)))
        
        else:
            num = int(np.hypot(x1_p-x0_p, y1_p-y0_p))
            x_cut, y_cut = np.linspace(x0_p, x1_p, num), np.linspace(y0_p, y1_p, num)
        
            # Extract the values along the line
            zi = z.T[x_cut.astype(int), y_cut.astype(int)]
        
        return zi

    
    def _update_slider(self,val,xy='x',ind=0):
    
        if xy=='x':
            self.x_points[ind] = self.x_sliders[ind].val
            if self.bfix.value_selected=='verti':
                self.x_points[1-ind] = self.x_sliders[ind].val
        else:
            self.y_points[ind] = self.y_sliders[ind].val  
            if self.bfix.value_selected=='hori':
                self.y_points[1-ind] = self.y_sliders[ind].val
        self._set()
        
    
    def _update_xsstart(self,val):
        self._update_slider(val,xy='x',ind=0)
        
    def _update_ysstart(self,val):
        self._update_slider(val,xy='y',ind=0)
    
    def _update_xsend(self,val):
        self._update_slider(val,xy='x',ind=1)
        
    def _update_ysend(self,val):
        self._update_slider(val,xy='y',ind=1)
    
    def _update_text(self,expression,xy='x',ind=0):
        
        if xy=='x':
            self.x_points[ind] = float(self.x_textboxes[ind].text)
            self.x_sliders[ind].eventson=False
            self.x_sliders[ind].set_val(self.x_points[ind])
            self.x_sliders[ind].eventson=True
            if self.bfix.value_selected=='verti':
                self.x_points[1-ind] = float(self.x_textboxes[ind].text)
                self.x_sliders[1-ind].eventson=False
                self.x_sliders[1-ind].set_val(self.x_points[1-ind])
                self.x_sliders[1-ind].eventson=True
        else:
            self.y_points[ind] = float(self.y_textboxes[ind].text)
            self.y_sliders[ind].eventson=False
            self.y_sliders[ind].set_val(self.y_points[ind])
            self.y_sliders[ind].eventson=True
            if self.bfix.value_selected=='hori':
                self.y_points[1-ind] = float(self.y_textboxes[ind].text)
                self.y_sliders[1-ind].eventson=False
                self.y_sliders[1-ind].set_val(self.y_points[1-ind])
                self.y_sliders[1-ind].eventson=True
        slope = (self.y_points[1]-self.y_points[0])/(self.x_points[1]-self.x_points[0])
        self.slope_text.set_text(f"Slope: {slope:4.3}")
        self._set()
    
    def _update_xtstart(self,expression):
        self._update_text(expression,xy='x',ind=0)
        
    def _update_ytstart(self,expression):
        self._update_text(expression,xy='y',ind=0)
    
    def _update_xtend(self,expression):
        self._update_text(expression,xy='x',ind=1)
    
    def _update_ytend(self,expression):
        self._update_text(expression,xy='y',ind=1)
    
    
    def _reset(self,event):

        #reset the values
        self.x_points = [self.x0,self.x1]
        self.y_points = [self.y0,self.y1]
        for i in np.arange(2):
          self.y_sliders[i].reset()
          self.x_sliders[i].reset()   
        self.bfix.set_active(0)
        self._set()
        
    
    def _set(self):
        
        x_points, y_points = self.x_points, self.y_points
        
        self.l0.set_xdata(x_points[0])
        self.l0.set_ydata(y_points[0])
        self.l1.set_xdata(x_points[1])
        self.l1.set_ydata(y_points[1])
        self.m.set_ydata(y_points)
        self.m.set_xdata(x_points)
        
        zi = self._make_linecut(x_points[0],y_points[0],x_points[1],y_points[1])
        linecut_energy = 1000*(self.lever_x*(x_points[1]-x_points[0])+self.lever_y*(y_points[1]-y_points[0])) #meV
        
        self.linecut.set_ydata(zi)
        self.linecut.set_xdata(np.linspace(0,linecut_energy,np.size(zi)))
        
        self.line_s.set_ydata(zi[0])
        self.line_s.set_xdata(0)
        self.line_f.set_ydata(zi[-1])
        self.line_f.set_xdata(linecut_energy)
        self.axes[1].set_xlim([-0.05*linecut_energy,1.05*linecut_energy])
        
        self.linecut_energy = linecut_energy
        self.zi = zi 
        # redraw canvas while idle      
        self.fig1.canvas.draw_idle()
        
    def _button_press_callback(self,event):
        'whenever a mouse button is pressed'

        if event.inaxes is None:
            return
        if event.button != 1:
            return
        #print(pind)
        self.pind = self._get_ind_under_point(event)
        
    def _button_release_callback(self,event):
        'whenever a mouse button is released'
        
        x_points, y_points = self.x_points, self.y_points
        
        if event.button != 1:
            return
        slope = (y_points[1]-y_points[0])/(x_points[1]-x_points[0])
        self.slope_text.set_text(f"Slope: {slope:4.3}")
        self.pind = None
        for i in range(2):
            self.x_textboxes[i].eventson=False
            self.x_textboxes[i].set_val(f"{self.x_points[i]:4.3f}")
            self.x_textboxes[i].eventson=True
            self.y_textboxes[i].eventson=False
            self.y_textboxes[i].set_val(f"{self.y_points[i]:4.3f}")
            self.y_textboxes[i].eventson=True
    
        
    
    def _get_ind_under_point(self,event):
        'get the index of the vertex under point if within epsilon tolerance'
        x_points, y_points = self.x_points, self.y_points
    
        # display coords
        #print('display x is: {0}; display y is: {1}'.format(event.x,event.y))
        t = self.axes[0].transData.inverted()
        tinv = self.axes[0].transData 
        xy = t.transform([event.x,event.y])
        #print('data x is: {0}; data y is: {1}'.format(xy[0],xy[1]))
        xr = np.reshape(x_points,(np.shape(x_points)[0],1))
        yr = np.reshape(y_points,(np.shape(y_points)[0],1))
        xy_vals = np.append(xr,yr,1)
        xyt = tinv.transform(xy_vals)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.hypot(xt - event.x, yt - event.y)
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]
    
        #print(d[ind])
        if d[ind] >= self.epsilon:
            ind = None
        
        #print(ind)
        return ind
    
    def _motion_notify_callback(self,event):
        
        'on mouse movement'

        if self.pind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        
        #update yvals
        #print('motion x: {0}; y: {1}'.format(event.xdata,event.ydata))
        slope = (self.y_points[1]-self.y_points[0])/(self.x_points[1]-self.x_points[0])
        
        self.y_points[self.pind] = event.ydata
        self.x_points[self.pind] = event.xdata
    
            
        # update curve via sliders and draw
    
        self.y_sliders[self.pind].set_val(self.y_points[self.pind])
        self.x_sliders[self.pind].set_val(self.x_points[self.pind])
        if self.bfix.value_selected=='free':
            pass
        elif self.bfix.value_selected=='hori':
            self.y_sliders[1-self.pind].set_val(self.y_points[self.pind])
        elif self.bfix.value_selected=='verti':
            self.x_sliders[1-self.pind].set_val(self.x_points[self.pind])
        else:
            fix_vec = np.array([1,slope])
            fix_vec = fix_vec/np.sum((fix_vec**2))**0.5
            other_vec = np.array([(self.x_points[1-self.pind]-self.x_points[self.pind]),(self.y_points[1-self.pind]-self.y_points[self.pind])])
            new_points = np.sum((other_vec*fix_vec))*fix_vec+np.array([self.x_points[self.pind],self.y_points[self.pind]])
    
            self.x_sliders[1-self.pind].set_val(new_points[0])
            self.y_sliders[1-self.pind].set_val(new_points[1])
    
    
        self.fig1.canvas.draw_idle()
        
    def _calc_max(self,event):
        max_idx = np.argmax(self.zi)
        current_max = self.zi[max_idx]
        
        upper_HM_idx = int(max_idx+np.argmin(np.abs(self.zi[max_idx:]-current_max/2)))
        
        lowerbound_HM_search = max(0,max_idx+3*(max_idx-upper_HM_idx))
        
        lower_HM_idx = lowerbound_HM_search+int(np.argmin(np.abs(self.zi[lowerbound_HM_search:max_idx]-current_max/2)))
    
        
        x_linecut = np.linspace(0,self.linecut_energy,np.size(self.zi))
        
        FWHM = x_linecut[upper_HM_idx]-x_linecut[lower_HM_idx]
        
        self.max_line.set_xdata([x_linecut[lower_HM_idx],x_linecut[upper_HM_idx]])
        self.max_line.set_ydata([current_max,current_max])
        
        self.left_HM_line.set_xdata([x_linecut[lower_HM_idx],x_linecut[lower_HM_idx]])
        self.left_HM_line.set_ydata([-0.2*current_max,1.2*current_max])
        
        self.right_HM_line.set_xdata([x_linecut[upper_HM_idx],x_linecut[upper_HM_idx]])
        self.right_HM_line.set_ydata([-0.2*current_max,1.2*current_max])
        
        self.max_text.set_text(f"Max: {current_max:4.3}\nFWHM: {FWHM:4.3}mV")
        
        self.fig1.canvas.draw_idle()
    
    def make_plot(self,vmax=None):
        """
        Make the manual-fitting window and GUI
        Parameters
        ----------
        vmax : float, optional
            Cut-off the maximum value of the plot. If None, all values will be considered 
        """
        
        x_points, y_points = self.x_points, self.y_points
        x,y,z = self.x,self.y,self.z
        zi = self.zi
        
        # axes=[]
        self.fig1, self.axes = plt.subplots(ncols=2)
        # fig1, ax1 = plt.subplots(ncols=1)
        # axes.append(ax0)
        # axes.append(ax1)
        self.fig1.subplots_adjust(right=0.7)
        im = self.axes[0].imshow(z,extent=[np.min(x),np.max(x),np.min(y),np.max(y)],aspect='auto',vmax=vmax)
        self.m, = self.axes[0].plot([self.x0, self.x1], [self.y0, self.y1], 'r-')
        self.l0, = self.axes[0].plot([self.x0], [self.y0], 'yo')
        self.l1, = self.axes[0].plot([self.x1], [self.y1], 'ko')
        self.axes[0].set_xlabel('SL (V)')
        self.axes[0].set_ylabel('BL (V)')

        plt.colorbar(im,ax=self.axes[0])

        self.linecut_energy = 1000*(self.lever_x*(x_points[1]-x_points[0])+self.lever_y*(y_points[1]-y_points[0]))
        self.linecut, = self.axes[1].plot(np.linspace(0,self.linecut_energy,np.size(zi)),zi,'r')
        self.line_s, = self.axes[1].plot(0,zi[0],'yo')
        self.line_f, = self.axes[1].plot(self.linecut_energy,zi[-1],'ko')

        self.max_line, = self.axes[1].plot([10*self.linecut_energy,10*self.linecut_energy],[0,0],'b')
        self.left_HM_line, = self.axes[1].plot([10*self.linecut_energy,10*self.linecut_energy],[0,1],'b')
        self.right_HM_line, = self.axes[1].plot([10*self.linecut_energy,10*self.linecut_energy],[0,1],'b')

        self.axes[1].set_xlim([-0.05*self.linecut_energy,1.05*self.linecut_energy])
        self.axes[1].set_ylim([np.min(z),np.max(z)])
        self.axes[1].set_ylabel(r'$G (G_0=2e^2/h)$')
        self.axes[1].set_xlabel('E_dot (meV assuming some lever arm)')

        self.y_sliders = []
        self.x_sliders = []

        self.x_textboxes = []
        self.y_textboxes = []

        for i in np.arange(2):

            axamp_x = plt.axes([0.74, 0.8-(i*0.1), 0.12, 0.02])
            axamp_y = plt.axes([0.74, 0.75-(i*0.1), 0.12, 0.02])
            # Slider
            xs = Slider(axamp_x, 'x{0}'.format(i), min(x), max(x), valinit=x_points[i])
            ys = Slider(axamp_y, 'y{0}'.format(i), min(y), max(y), valinit=y_points[i])
            
            xs.valtext.set_visible(False)
            ys.valtext.set_visible(False)
            
            self.x_sliders.append(xs)
            self.y_sliders.append(ys)
            
            axamp_x = plt.axes([0.87, 0.8-(i*0.1), 0.08, 0.03])
            axamp_y = plt.axes([0.87, 0.75-(i*0.1), 0.08, 0.03])
            
            xt = TextBox(axamp_x,label=None, initial=f"{x_points[i]:4.3f}")
            yt = TextBox(axamp_y,label=None, initial=f"{y_points[i]:4.3f}")
            
            self.x_textboxes.append(xt)
            self.y_textboxes.append(yt)
            

        self.x_sliders[0].on_changed(self._update_xsstart)
        self.y_sliders[0].on_changed(self._update_ysstart)
        self.x_sliders[1].on_changed(self._update_xsend)
        self.y_sliders[1].on_changed(self._update_ysend)

        self.x_textboxes[0].on_submit(self._update_xtstart)
        self.y_textboxes[0].on_submit(self._update_ytstart)
        self.x_textboxes[1].on_submit(self._update_xtend)
        self.y_textboxes[1].on_submit(self._update_ytend)
            

        axres = plt.axes([0.74, 0.8-((5)*0.05), 0.12, 0.02])
        self.bres = Button(axres, 'Reset')
        self.bres.on_clicked(self._reset)

        axres = plt.axes([0.74, 0.8-((8)*0.05), 0.12, 0.15])
        self.bfix = RadioButtons(axres, ['free','hori','verti','fixed'])

        # axinfo = plt.axes([0.86, 0.8-((8)*0.05), 0.12, 0.15])

        self.slope = (y_points[1]-y_points[0])/(x_points[1]-x_points[0])
        box_props = dict(boxstyle='round',facecolor='white')
        self.slope_text = axres.text(0,-0.5,  f"Slope: {self.slope:4.3}",bbox=box_props)

        axmax = plt.axes([0.74, 0.2, 0.15, 0.02])
        self.bmax = Button(axmax, 'Calc Max:')
        self.bmax.on_clicked(self._calc_max)
        max_val = 0.0
        FWHM = 0.0
        box_props = dict(boxstyle='round',facecolor='white')
        self.max_text = axres.text(0,-1,  f"Max: {max_val:4.3}G0\nFWHM: {1000*FWHM:4.3}mV",bbox=box_props)

        self.fig1.canvas.mpl_connect('button_press_event', self._button_press_callback)
        self.fig1.canvas.mpl_connect('button_release_event', self._button_release_callback)
        self.fig1.canvas.mpl_connect('motion_notify_event', self._motion_notify_callback)
    
