#!/usr/bin/env python3
"""Plot the results from lattice-sir.py
"""

import argparse
import logging
import os
from os.path import join as pjoin
from logging import debug, info

import datetime
import numpy as np
import pandas as pd
import scipy
import scipy.stats

import plotly
import plotly.graph_objects as go
from ipywidgets import widgets
import plotly.express as px

##########################################################
def read_niterations(outdir):
    """Read number of iterations for in each folder in @outdir

    Args:
    outdir(str): results directory containing the subolders

    Returns:
    dict: folder names as keys and counts as values
    """

    counts = {}

    for expidx in os.listdir(outdir):
        if not os.path.isdir(pjoin(outdir, expidx)): continue
        summarypath = pjoin(outdir, expidx, 'sir.csv')
        counts[expidx] = sum(1 for line in open(summarypath))

    return counts

##########################################################
def get_inverse_map(mydict):
    return {v: k for k, v in mydict.items()}

##########################################################
def remap_to_categorical_data(df, cols):
    tickslabels = {}
    for col in cols:
        vals = sorted(df[col].unique())
        tickslabels[col] = vals
        aux = dict(enumerate(vals))
        aux = get_inverse_map(aux)
        df[col] = df[col].map(aux)
    return df, tickslabels

##########################################################
def plot_parallel_coordinates(expsdf, colslabels, categcols, tickslabels, outdir):

    dimensions = []
    for col in categcols: # categorical columns
        colname = colslabels[col]
        plotcol = dict(
            label = colname,
            values = expsdf[col],
            tickvals = list(range(len(tickslabels[col]))),
            ticktext = tickslabels[col])
        dimensions.append(plotcol)

    dimensions.append(
        dict(label = 'Convergence time',
             values = expsdf['t'],
    ))

    fig = go.Figure(data=go.Parcoords(
        line_color='blue',
        dimensions = dimensions
    )
                    )
    plotpath = pjoin(outdir, 'parallel.html')
    plotly.offline.plot(fig, filename=plotpath, auto_open=False)

##########################################################
def plot_parallel(resdir, outdir):
    dfs = []
    for i, resdir in enumerate(resdir):
        expspath = pjoin(resdir, 'exps.csv')
        df = pd.read_csv(expspath, index_col='expidx')
        niterations = read_niterations(resdir)
        niterations = pd.Series(niterations, index=df.index, dtype=int)
        df['t'] = niterations
        dfs.append(df)

    df = pd.concat(dfs)
    colslabels = dict(topologymodel = 'topology',
                      lathoroidal = 'lattice-thoroidal',
                      beta = 'beta',
                      gamma = 'gamma',
                      gaussianstd = 'gradients dispersion',
                      t = 'convergence time',
                      )
    df = df[colslabels.keys()]
    categcols = list(colslabels.keys())
    categcols.remove('t')
    expsdf, tickslabels = remap_to_categorical_data(df, categcols)
    plot_parallel_coordinates(expsdf, colslabels, categcols, tickslabels, outdir)

##########################################################
def plot_sir_all(resdir, outdir, nseeds=3):

    resdir = resdir[0]
    expspath = pjoin(resdir, 'exps.csv')
    df = pd.read_csv(expspath, index_col='expidx')

    import matplotlib.pyplot as plt
    plotrows = int(df.shape[0]/nseeds)
    fig, ax = plt.subplots(plotrows, nseeds, figsize=(8, plotrows*2))

    mycols = list(df.columns)
    mycols.remove('randomseed')
    mycols.remove('hostname')
    df_sorted = df.sort_values(mycols)
    plt.tight_layout(pad=2.5, h_pad=3.0, w_pad=0.5)

    for j, expidx in enumerate(df_sorted.index):
        if not os.path.isdir(pjoin(resdir, expidx)): continue
        summarypath = pjoin(resdir, expidx, 'sir.csv')
        aux = pd.read_csv(summarypath)
        row = j//nseeds
        col = j%nseeds
        ax[row, col].plot(aux.t, aux.S, label='S')
        ax[row, col].plot(aux.t, aux.I, label='I')
        ax[row, col].plot(aux.t, aux.R, label='R')
        ax[row, col].legend(fontsize='small')
        ax[row, col].set_xlim(0, 500)
        v = df_sorted.loc[expidx]
        title = 'topol:{},avgdegree:{},latt-thorus:{},\nbeta:{},gamma:{},gradspread:{}'. \
            format(v.topologymodel, v.avgdegree, v.lathoroidal, v.beta, v.gamma,
                   v.gaussianstd)
        ax[row, col].set_title(title, fontsize='small')
    plt.savefig(pjoin(outdir, 'sir_all.pdf'))

##########################################################
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('resdir', nargs='+',
                        help='Directory(ies) containing the output from lattice-sir')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
                        datefmt='%Y%m%d %H:%M', level=logging.DEBUG)

    outdir = '/tmp'
    # plot_parallel(args.resdir, outdir)
    plot_sir_all(args.resdir, outdir)

if __name__ == "__main__":
    main()

