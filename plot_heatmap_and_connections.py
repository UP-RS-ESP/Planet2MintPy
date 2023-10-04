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
import os
    
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
    
    agg = pd.DataFrame({"date": pd.concat([df.date0, df.date1]), "dx_iqr": pd.concat([df.dx_iqr, df.dx_iqr]), "dy_iqr": pd.concat([df.dy_iqr, df.dy_iqr])})
    agg = agg.groupby("date").aggregate("median").reset_index()
    

    merge = merge.merge(agg, left_on = "date0", right_on = "date", suffixes=("", "_agg_ref"))
    merge = merge.merge(agg, left_on = "date1", right_on = "date", suffixes=("", "_agg_sec"))

    fig, ax = plt.subplots(1,2, figsize = (12,5))
    cmap = plt.get_cmap('viridis')
    norm = Normalize(vmin = vmin, vmax = vmax)
    for d0, d1, c0, c1, s in zip(merge.date0, merge.date1, merge.dx_iqr_agg_ref, merge.dx_iqr_agg_sec, merge["dx_iqr"]):
        x_values = [d0, d1]
        y_values = [c0, c1]
        color = cmap(norm(s))
        ax[0].plot(x_values, y_values, color=color)
    ax[0].scatter(agg.date, agg.dx_iqr)
    ax[0].set_xlabel("Date")
    ax[0].set_ylabel("median dx IQR ")
    for d0, d1, c0, c1, s in zip(merge.date0, merge.date1, merge.dy_iqr_agg_ref, merge.dy_iqr_agg_sec, merge["dy_iqr"]):
        x_values = [d0, d1]
        y_values = [c0, c1]
        color = cmap(norm(s))
        ax[1].plot(x_values, y_values, color=color)
    ax[1].scatter(agg.date, agg.dy_iqr)
    ax[1].set_xlabel("Date")
    ax[1].set_ylabel("median dy IQR ")
    
    ax[0].set_ylim(0,1.1)       
    ax[1].set_ylim(0,1.1)
    ax[0].grid()
    ax[1].grid()
    sm = ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, label = "dx_iqr", ax = ax[0]) 
    plt.colorbar(sm, label = "dy_iqr", ax = ax[1]) 
        
    plt.tight_layout()
    
    # fig, ax = plt.subplots()
    # cmap = plt.get_cmap('viridis')
    # norm = Normalize(vmin = vmin, vmax = vmax)
    # for d0, d1, c0, c1, s in zip(merge.date0, merge.date1, merge.count_x, merge.count_y, merge[color_by]):
    #     x_values = [d0, d1]
    #     y_values = [c0, c1]
    #     color = cmap(norm(s))
    #     ax.plot(x_values, y_values, color=color)
    # ax.scatter(conns.date, conns["count"])
    # sm = ScalarMappable(cmap=cmap, norm=norm)
    # plt.colorbar(sm, label = color_by, ax = ax) 
    
        
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

def get_scene_id(fn):
    
    #extract the scene id from a PS scene filename
    #assumes the filename still begins with the scene ID (should be default when downloading data)
    
    _, fn = os.path.split(fn) 
    
    #determine processing level of scenes
    if "_1B_" in fn:
        level = 1
    elif "_3B_" in fn:
        level = 3
    else:
        print("Could not determine processing level of the data. Make sure that either _1B_ or _3B_ is included in the filename of your scene.")
        return
    
    if fn.split("_").index(f"{level}B") == 4: #PSB.SD case
        scene_id = "_".join(fn.split("_")[0:4])
    elif fn.split("_").index(f"{level}B") == 3: #PS2 case
        scene_id = "_".join(fn.split("_")[0:3])
    else: 
        print("Couldn't guess the instrument type. Have you modifies filenames?")
        return
    return scene_id

        
def get_date(scene_id):
    
    #strip the time from th PS scene id
    
    return datetime.strptime(scene_id[0:8], "%Y%m%d")



for aoi in [3,4,5,6,7,9,10]:
    df = pd.read_csv(f"/home/ariane/Documents/Project3/PlanetScope_Data/aoi{aoi}/all_scenes/matches_by_group_PS2.csv")
    df2 = pd.read_csv(f"/home/ariane/Documents/Project3/PlanetScope_Data/aoi{aoi}/all_scenes/matches_by_group_PSB.SD.csv")
    
    df = pd.concat([df, df2]).reset_index(drop = True)
    df["id_ref"] = df.ref.apply(get_scene_id)
    df["id_sec"] = df.sec.apply(get_scene_id)
    df["date0"] = df.id_ref.apply(get_date)
    df["date1"] = df.id_sec.apply(get_date)
    
    
    conns = pd.concat([df.date0, df.date1]).value_counts().reset_index().rename(columns = {"index":"date", 0:"count"})
    merge = df.merge(conns, left_on='date0', right_on='date')
    merge = merge.merge(conns, left_on='date1', right_on='date')
    
    
    fig, ax = plt.subplots(1,1, figsize = (8,5))
    cmap = plt.get_cmap('viridis')
    for d0, d1, c0, c1 in zip(merge.date0, merge.date1, merge.count_x, merge.count_y):
        x_values = [d0, d1]
        y_values = [c0, c1]
        ax.plot(x_values, y_values, c = "royalblue")
    ax.scatter(conns.date, conns["count"],  c = "royalblue")
    ax.set_ylim(0, max(conns["count"])+1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of connections")
    ax.set_title(f"aoi{aoi}")
    plt.grid()
    
    plt.savefig(f"/home/ariane/Documents/Project3/connections/connections_aoi{aoi}.png", dpi = 300)
