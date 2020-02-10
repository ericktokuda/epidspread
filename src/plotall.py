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
        summarypath = pjoin(outdir, expidx, 'ntransmperepoch.csv')
        counts[expidx] = sum(1 for line in open(summarypath))
        # lastline = subprocess.check_output(['tail', '-1', summarypath]).decode('utf-8')
        # aux = lastline.split(',')[-2]
        # ninfected[expidx] = int(aux)

    return counts

##########################################################
def read_infected_ratios(outdir):
    infrat = {}

    for expidx in os.listdir(outdir):
        if not os.path.isdir(pjoin(outdir, expidx)): continue
        summarypath = pjoin(outdir, expidx, 'ntransmperepoch.csv')
        df = pd.read_csv(summarypath)
        n = float(df.iloc[0].S + df.iloc[0].I + df.iloc[0].R )
        r = df.iloc[-1].R / n
        infrat[expidx] = r

    return infrat
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

    dimensions.append(
        dict(label = 'Infected ratio',
             values = expsdf['inf'],
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
        inf = read_infected_ratios(resdir)
        inf = pd.Series(inf, index=df.index, dtype=float)
        df['inf'] = inf

        dfs.append(df)

    df = pd.concat(dfs)
    colslabels = dict(topologymodel = 'topology',
                      avgdegree = 'avg.degree',
                      lathoroidal = 'lattice-thoroidal',
                      wsrewiring = 'WS rew. prob.',
                      wxalpha = 'WX alpha',
                      mobilityratio = 'mob. ratio',
                      beta = 'beta',
                      gamma = 'gamma',
                      gaussianstd = 'gradients dispersion',
                      t = 'convergence time',
                      inf = 'infected ratio',
                      )

    df = df[colslabels.keys()]
    categcols = list(colslabels.keys())
    categcols.remove('t')
    categcols.remove('inf')
    expsdf, tickslabels = remap_to_categorical_data(df, categcols)
    plot_parallel_coordinates(expsdf, colslabels, categcols, tickslabels, outdir)

##########################################################
def filter_exps_df(df, dffilter):
    for k, v in dffilter.items():
        df = df[(df[k].isin(v))]
        info('Filtering by k:{} resulted in {} rows'.format(k, df.shape[0]))
    return df

##########################################################
def plot_recoveredrate_vs_beta_all(resdir, outdir):
    expspath = pjoin(resdir[0], 'exps.csv')
    df = pd.read_csv(expspath, index_col='expidx')
    df = df.sort_values(['topologymodel', 'beta', 'gamma', 'gaussianstd', 'avgdegree'])
    gammas = np.unique(df.gamma)
    for g in gammas:
        plot_recoveredrate_vs_beta(resdir, g, outdir)

##########################################################
def plot_recoveredrate_vs_beta(resdir, gamma, outdir):
    dffilter = {
        "nvertices" : [625],
        "avgdegree" : [4],
        "wxalpha" : [-1.0],
    }

    expspath = pjoin(resdir[0], 'exps.csv')
    df = pd.read_csv(expspath, index_col='expidx')
    df = df.sort_values(['topologymodel', 'beta', 'gamma', 'gaussianstd', 'avgdegree'])

    info('Before filering:')
    for col in df.columns:
        info('{}:{}'.format(col, np.unique(df[col])))

    df = filter_exps_df(df, dffilter)

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

                # input(aux3.index)
                for expidx in aux3.index:
                    # print('expidx:{}'.format(expidx))
                    # print('exppath:{}'.format(pjoin(resdir[0], expidx, 'summary.csv')))
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
            # print(data[data.gaussianstd==std_])
            # input()
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

##########################################################
def plot_waxman_all(resdir, outdir):
    dffilter = {
        "nvertices" : [625],
        "topologymodel" : ["wx"],
        "avgdegree" : [4],
        "gamma" :  [0.5],
    }

    expspath = pjoin(resdir[0], 'exps.csv')
    df = pd.read_csv(expspath, index_col='expidx')
    df = df.sort_values(['topologymodel', 'beta', 'gamma', 'gaussianstd', 'avgdegree'])
    df = filter_exps_df(df, dffilter)

    tops = np.unique(df.topologymodel)
    betas = np.unique(df.beta)
    stds = np.unique(df.gaussianstd)

    alphas = np.unique(df.wxalpha)
    fig, ax = plt.subplots(1, len(alphas), figsize=(6*len(alphas), 6))

    for a, col in zip(ax, [str(alpha).upper() for alpha in alphas]):
        a.set_title(col, size='x-large')

    cols = ['wxalpha', 'beta', 'gaussianstd', 'recmean', 'recstd']

    for j, alpha in enumerate(alphas):
        datadict = {k: [] for k in cols}
        aux = df[df.wxalpha == alpha]

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
                datadict['wxalpha'].append(alpha)
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

    plt.tight_layout()
    fig.suptitle('Total infected ratio vs Contagion rate', size='xx-large')
    plt.savefig(pjoin(outdir, '{}.pdf'.format('waxmans')))

##########################################################
def plot_ntransmissions_vs_gradient(resdir, expid, ax):
    expdir = pjoin(resdir, expid)
    gradpath = pjoin(expdir, 'attraction.csv')
    ntransmpath = pjoin(expdir, 'ntransmpervertex.csv')
    grads = pd.read_csv(gradpath).gradient
    ntransm = pd.read_csv(ntransmpath).ntransmission
    # print(ntransm, grads)
    ax.scatter(grads, ntransm)

##########################################################
def plot_ntransmissions_vs_gradient_all(resdir, outdir):
    dffilter = {
        "topologymodel" : ["wx"],
        "avgdegree" : [4],
        "beta" :  [0.6],
        "gamma" :  [0.5],
        "gaussianstd" : [0.15],
    }

    expspath = pjoin(resdir[0], 'exps.csv')
    df = pd.read_csv(expspath, index_col='expidx')
    df = df.sort_values(['topologymodel', 'beta', 'gamma', 'gaussianstd', 'avgdegree'])
    df = filter_exps_df(df, dffilter)

    alphas = sorted(np.unique(df.wxalpha))
    seeds = sorted(np.unique(df.randomseed))
    nalphas = len(alphas)
    nseeds = len(seeds)
    fig, ax = plt.subplots(nseeds, nalphas, squeeze=False, figsize=(6*nalphas, 6*nseeds))

    for i, seed in enumerate(seeds):
        for j, alpha in enumerate(alphas):
            dffilter = {'randomseed': [i], 'wxalpha': [alpha]}
            expid = filter_exps_df(df, dffilter).index.values[0]
            plot_ntransmissions_vs_gradient(resdir[0], expid, ax[i, j])

    fig.suptitle('Number of transmissions vs. Atraction on Waxman topologies\nROW: how far waxman is close to ER; COLUMNS: realizations', size=60)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(pjoin(outdir, 'ntransmissions.png'))

##########################################################
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
    # return
    # plot_sir_all(args.resdir, outdir)
    # plot_measures_luc(args.resdir, outdir)
    # plot_pca(args.resdir, outdir)
    # plot_recoveredrate_vs_beta_all(args.resdir, outdir)
    # plot_waxman_all(args.resdir, outdir)
    plot_ntransmissions_vs_gradient_all(args.resdir, outdir)
    # plot_attraction('/tmp/del/aw0otrky/attraction.csv')

if __name__ == "__main__":
    main()

