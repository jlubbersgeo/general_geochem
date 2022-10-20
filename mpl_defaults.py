#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 23:35:32 2022

@author: jordanlubbers
"""

import matplotlib.pyplot as plt
import matplotlib
import os

# rebuild fonts list...for if you recently
# added a new ttf file.
# matplotlib.font_manager._rebuild()

#removes the matplotlib cache so fonts always work
#this is usually in the form of a json file with 
#"fontlist" somewhere in the filename.
cache_dir = matplotlib.get_cachedir()

#for windows paths
for file in os.listdir(cache_dir):
    if 'fontlist' in file:
        os.remove(cache_dir +'\\'+file)
        
        
        
        
# setting matplotlib defaults
p = plt.rcParams


# figure-wide aesthetics
p["figure.figsize"] = [6,4]
p["figure.edgecolor"] = "black"
p["figure.facecolor"] = "none"
p['figure.dpi'] = 100
p['savefig.dpi'] = 300
p['figure.titlesize'] = 24


#axes-level aesthetics
p["axes.linewidth"] = 1
p["axes.facecolor"] = "white"
p["axes.ymargin"] = 0.1
p["axes.spines.bottom"] = True
p["axes.spines.left"] = True
p["axes.spines.right"] = True
p["axes.spines.top"] = True
p["axes.labelsize"] = 20

#font stuff
p['font.family'] = 'sans-serif'
# on my computer good options for sans serif are
# Arial, Roboto Condensed, CMU Sans Serif, Fira Sans Condensed
p['font.sans-serif'] = ['CMU Sans Serif']
# computer modern for serif fonts
p['font.serif'] = ['CMU Serif']

#grid defaults
p["axes.grid"] = False
p["grid.color"] = "black"
p["grid.linewidth"] = 0.1

#x-tick customization
p["xtick.bottom"] = True
p["xtick.top"] = False
p["xtick.direction"] = "out"
p["xtick.major.size"] = 5
p["xtick.major.width"] = 1
p["xtick.minor.size"] = 3
p["xtick.minor.width"] = 0.5
p["xtick.minor.visible"] = True
p['xtick.labelsize']= 10

#y-tick customization
p["ytick.left"] = True
p["ytick.right"] = False
p["ytick.direction"] = "out"
p["ytick.major.size"] = 5
p["ytick.major.width"] = 1
p["ytick.minor.size"] = 3
p["ytick.minor.width"] = 0.5
p["ytick.minor.visible"] = True
p['ytick.labelsize']= 10


#marker customization
p["lines.linewidth"] = 1.5
p["lines.marker"] = ""
p["lines.markeredgewidth"] = 0.5
p["lines.markeredgecolor"] = "k"
p["lines.markerfacecolor"] = "auto"
p["lines.markersize"] = 7


#helper function for removing the top 
#and right spines for a simple looking plot
def left_bottom_axes(ax):
  
    for spine in ['top','right']:
        ax.spines[spine].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_facecolor('white')

#helper function for removing the top 
#and right spines for a simple looking plot
def bottom_only_axes(ax):
  
    for spine in ['top','right','left']:
        ax.spines[spine].set_visible(False)
    ax.set_yticks([])
    ax.get_xaxis().tick_bottom()
    ax.set_facecolor('white')
    
    
#helper for labeling subplots
from matplotlib.offsetbox import AnchoredText
import string

def label_subplots(axes,location,fontsize = 14,alpha = .5):
    if len(axes.shape) > 1:
        axes = axes.ravel()
    letters = list(string.ascii_uppercase)

    for a,letter in zip(axes,letters):
        at = AnchoredText("{}".format(letter), prop=dict(size=fontsize),frameon=True, loc= location)
        at_noletters = AnchoredText("{}".format(letter), prop=dict(size=fontsize,color = 'white'),frameon=True, loc=location)

        at_noletters.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        at.patch.set_linewidth(2)

        at_noletters.patch.set_facecolor('white')
        at.patch.set_facecolor('none')

        at_noletters.patch.set_alpha(alpha)
        a.add_artist(at_noletters)
        a.add_artist(at)
    

    
