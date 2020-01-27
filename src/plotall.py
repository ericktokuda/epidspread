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
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
    # from numpy.linalg import eig

import plotly
import plotly.graph_objects as go
from ipywidgets import widgets
import plotly.express as px
import subprocess

##########################################################
def read_niterations(outdir):
    """Read number of iterations for in each folder in @outdir

    Args:
    outdir(str): results directory containing the subolders

    Returns:
    dict: folder names as keys and counts as values
    """

    counts = {}
    # ninfected = {}

    for expidx in os.listdir(outdir):
        if not os.path.isdir(pjoin(outdir, expidx)): continue
        summarypath = pjoin(outdir, expidx, 'transmcount.csv')
        counts[expidx] = sum(1 for line in open(summarypath))
        # lastline = subprocess.check_output(['tail', '-1', summarypath]).decode('utf-8')
        # aux = lastline.split(',')[-2]
        # ninfected[expidx] = int(aux)

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
                      avgdegree = 'avg.degree',
                      lathoroidal = 'lattice-thoroidal',
                      wsrewiring = 'WS rew. prob.',
                      mobilityratio = 'mob. ratio',
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
def filter_exps_df(df, nagentspervertex=None, avgdegree=None, lathoroidal=None,
                   wsrewiring=None, wxalpha=None, mobilityratio=None, gamma=None):
    df = df[(df.nagentspervertex == nagentspervertex)]
    info('Filtering by nagentspervertex:{} resulted in {} rows'.format(nagentspervertex,
                                                                 df.shape[0]))

    df = df[(df.avgdegree == avgdegree) |
            ((df.topologymodel == 'la') & (df.avgdegree == 4))]
    # df = df[(df.avgdegree == avgdegree)]
    info('Filtering by avgdegree:{} (except lattice) resulted in {} rows'.format(avgdegree,
                                                                 df.shape[0]))

    df = df[(df.lathoroidal == -1) | (df.lathoroidal == 0) ]
    info('Filtering by lathoroidal:{} resulted in {} rows'.format(lathoroidal,
                                                                 df.shape[0]))

    # print(np.unique(df.topologymodel))
    df = df[(df.wsrewiring == -1) | (df.wsrewiring == wsrewiring) ]
    info('Filtering by wsrewiring:{} resulted in {} rows'.format(wsrewiring,
                                                                 df.shape[0]))

    df = df[(df.wxalpha == -1) | (df.wxalpha == wxalpha) ]
    info('Filtering by wxalpha:{} resulted in {} rows'.format(wxalpha,
                                                                 df.shape[0]))
    # print(np.unique(df.topologymodel))

    df = df[(df.mobilityratio == mobilityratio)]
    info('Filtering by mobilityratio:{} resulted in {} rows'.format(mobilityratio,
                                                                 df.shape[0]))

    df = df[(df.gamma == gamma)]
    info('Filtering by gamma:{} resulted in {} rows'.format(gamma, df.shape[0]))
    return df

##########################################################
def plot_recoveredrate_vs_beta_gammas(resdir, outdir):
    # gamma = 0.3
    expspath = pjoin(resdir[0], 'exps.csv')
    df = pd.read_csv(expspath, index_col='expidx')
    df = df.sort_values(['topologymodel', 'beta', 'gamma', 'gaussianstd', 'avgdegree'])
    gammas = np.unique(df.gamma)
    for g in gammas:
        plot_recoveredrate_vs_beta(resdir, g, outdir)

def plot_recoveredrate_vs_beta(resdir, gamma, outdir):
    nagentspervertex = 1
    avgdegree = 6
    lathoroidal = -1
    wsrewiring = 0.001
    wxalpha = 0.2
    mobilityratio = -1.0

    expspath = pjoin(resdir[0], 'exps.csv')
    df = pd.read_csv(expspath, index_col='expidx')
    df = df.sort_values(['topologymodel', 'beta', 'gamma', 'gaussianstd', 'avgdegree'])
    df = filter_exps_df(df, nagentspervertex=nagentspervertex, avgdegree=avgdegree,
                   lathoroidal=lathoroidal, wsrewiring=wsrewiring, wxalpha=wxalpha,
                   mobilityratio=mobilityratio, gamma=gamma)

    tops = np.unique(df.topologymodel)
    betas = np.unique(df.beta)
    stds = np.unique(df.gaussianstd)

    fig, ax = plt.subplots(1, len(tops), figsize=(6*len(tops), 6))

    for a, col in zip(ax, [str(t).upper() for t in tops]):
        a.set_title(col, size='x-large')

    cols = ['topologymodel', 'beta', 'gaussianstd', 'recmean', 'recstd']

    for j, top in enumerate(tops):
        datadict = {k: [] for k in cols}
        aux = df[df.topologymodel == top]

        valid = [] # filtering incomplete execution

        for b in betas:
            aux2 = aux[aux.beta == b]

            for std_ in stds:
                aux3 = aux2[aux2.gaussianstd == std_]

                nruns = aux3.shape[0]
                recoveredratios = []

                for expidx in aux3.index:
                    if not os.path.exists(pjoin(resdir[0], expidx, 'summary.csv')):
                        continue

                    valid.append(expidx)

                    aux4 = pd.read_csv(pjoin(resdir[0], expidx, 'ntransmperepoch.csv'))
                    n = float(aux4.iloc[0].S + aux4.iloc[0].I + aux4.iloc[0].R )
                    r = aux4.iloc[-1].R / n
                    recoveredratios.append(r)

                #append row
                datadict['topologymodel'].append(top)
                datadict['beta'].append(b)
                datadict['gaussianstd'].append(std_)
                datadict['recmean'].append(np.mean(recoveredratios))
                datadict['recstd'].append(np.std(recoveredratios))
        
        data = pd.DataFrame.from_dict(datadict)

        # Generating colormap
        categories = np.unique(data.gaussianstd)
        colors = np.linspace(0, 1, len(categories))
        colordict = dict(zip(categories, colors))
        mycmap = aux.gaussianstd.apply(lambda x: colordict[x])

        colors_ = ['#7fc97f','#beaed4','#fdc086','#ffff99',
                   '#386cb0','#f0027f','#bf5b17','#666666']

        for ii, std_ in enumerate(stds):
            ax[j].errorbar(data[data.gaussianstd==std_].beta,
                       data[data.gaussianstd==std_].recmean,
                       yerr=data[data.gaussianstd==std_].recstd,
                       marker='o', c=colors_[ii], label=str(std_),
                       alpha=0.75)

        ax[j].legend(title='Gaussian std')
        ax[j].set_xlim(left=0, right=1)
        ax[j].set_ylim(bottom=0, top=1)
    fig.suptitle('Total infected ratio vs Contagion rate', size='xx-large')
    plt.savefig(pjoin(outdir, '{}.pdf'.format(gamma)))

def plot_attraction(attractionpath, outpath='/tmp/attraction.pdf'):
    """Plot map of attraction according to the @attractionpath csv file

    Args:
    attractionpath(str): path to the csv file
    """
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import pandas as pd

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    df = pd.read_csv(attractionpath)
    ax.plot_trisurf(df.x.values, df.y.values, df.gradient.values, cmap=cm.coolwarm,)
    plt.savefig(outpath)

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
    # plot_sir_all(args.resdir, outdir)
    # plot_measures_luc(args.resdir, outdir)
    # plot_pca(args.resdir, outdir)
    plot_recoveredrate_vs_beta_gammas(args.resdir, outdir)
    # plot_attraction('/tmp/del/aw0otrky/attraction.csv')

if __name__ == "__main__":
    main()

