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

def get_inverse_map(mydict):
    return {v: k for k, v in mydict.items()}

def remap_to_categorical_data(df, cols):
    tickslabels = {}
    for col in cols:
        vals = sorted(df[col].unique())
        tickslabels[col] = vals
        aux = dict(enumerate(vals))
        aux = get_inverse_map(aux)
        df[col] = df[col].map(aux)
    return df, tickslabels

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

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('resdir', nargs='+',
                        help='Directory(ies) containing the output from lattice-sir')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
                        datefmt='%Y%m%d %H:%M', level=logging.DEBUG)

    outdir = '/tmp'

    # topologymap = {'erdos': 0, 'lattice': 1}
    topologynames = sorted(['erdos', 'lattice'])
    layoutnames = sorted(['grid', 'fr', 'kk'])
    erdosavgdegrnames = ['DotNotApply', '1', '4', '10', 'Fully']
    # layoutmap = {'fr': 0, 'grid': 1, 'kk': 2}

    dfs = []
    for i, resdir in enumerate(args.resdir):
        expspath = pjoin(resdir, 'exps.csv')
        df = pd.read_csv(expspath, index_col='expidx')
        niterations = read_niterations(resdir)
        niterations = pd.Series(niterations, index=df.index, dtype=int)
        df['t'] = niterations
        # df['topologymodel'] = df.topologymodel.map(topologymap)
        # df['layoutmodel'] = df.layoutmodel.map(layoutmap)
        dfs.append(df)

    df = pd.concat(dfs)
    colslabels = {'topologymodel': 'topology',
                'layoutmodel': 'spatiality',
                'erdosavgdegree': 'erdos-avgdegr',
                'latticethoroidal': 'lattice-thoroidal',
                'beta': 'beta',
                'gamma': 'gamma',
                'gaussianstds': 'gradients dispersion',
                't': 'convergence time',
                }
    df = df[colslabels.keys()]
    categcols = list(colslabels.keys())
    categcols.remove('t')
    expsdf, tickslabels = remap_to_categorical_data(df, categcols)
    plot_parallel_coordinates(expsdf, colslabels, categcols, tickslabels, outdir)

if __name__ == "__main__":
    main()

