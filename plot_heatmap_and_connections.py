#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ariane Mueting and Bodo Bookhagen

"""


import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

    
def plot_network(df, color_by = "dx_iqr", vmin = 0, vmax = 1):
    if df.date0.dtype == "O":
        df.date0 = pd.to_datetime(df.date0)  
    if df.date1.dtype == "O":
        df.date1 = pd.to_datetime(df.date1)  
    df["dx_iqr"] = df.dx_p75-df.dx_p25
    df["dy_iqr"] = df.dy_p75-df.dy_p25
    df["dt"] = df.date1 - df.date0   
    conns = pd.concat([df.date0, df.date1]).value_counts().reset_index().rename(columns = {"index":"date", 0:"count"})
    merge = df.merge(conns, left_on='date0', right_on='date')
    merge = merge.merge(conns, left_on='date1', right_on='date')
    
    
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin = vmin, vmax = vmax)
    for d0, d1, c0, c1, s in zip(merge.date0, merge.date1, merge.count_x, merge.count_y, merge[color_by]):
        x_values = [d0, d1]
        y_values = [c0, c1]
        color = cmap(norm(s))
        ax.plot(x_values, y_values, color=color)
    ax.scatter(conns.date, conns["count"])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, label = color_by, ax = ax) 
    
        
def plot_heatmap(df):   
    if df.date0.dtype == "O":
        df.date0 = pd.to_datetime(df.date0)  
    if df.date1.dtype == "O":
        df.date1 = pd.to_datetime(df.date1)  

    df = df.sort_values(by=['date0'])
    df["dx_iqr"] = df.dx_p75-df.dx_p25
    df["dy_iqr"] = df.dy_p75-df.dy_p25
    df["dt"] = df.date1 - df.date0
    
    # fig, ax = plt.subplots(1,2, figsize = (12,5))
    # ax[0].scatter(df['dt'].dt.days, df.dx_iqr, s = 2)
    # ax[0].set_ylim(0,1.5)
    # ax[1].scatter(df['dt'].dt.days, df.dy_iqr, s = 2)
    # ax[1].set_ylim(0,1.5)

    groups = df.group.unique() #TODO: remove groups
    
    for g in groups:   
        gdf = df.loc[df.group == g]

        #heatmap
        fig, ax = plt.subplots(1,2, figsize = (12,5))
    
        for i, what in enumerate(["dx_iqr", "dy_iqr"]):
            pivot_df = gdf.pivot_table(index='date1', columns='date0', values=what, aggfunc='mean')
            hm_data = pivot_df.to_numpy()
            im = ax[i].imshow(hm_data, cmap='viridis', aspect='auto', origin='lower', vmin = 0, vmax = 1.5)
            plt.colorbar(im, label=what, ax = ax[i])
            xlab = [datetime.strftime(i, "%d.%m.%Y") for i in list(pivot_df.columns)]
            ylab = [datetime.strftime(i, "%d.%m.%Y") for i in list(pivot_df.index)]
            ax[i].set_xticks(np.arange(len(xlab)), labels=xlab)
            ax[i].set_yticks(np.arange(len(ylab)), labels=ylab)
            ax[i].set_title(g)
        
            # Rotate the tick labels and set their alignment.
            plt.setp(ax[i].get_xticklabels(), rotation=90, ha="right",
                     rotation_mode="anchor")
            
        plt.tight_layout()
