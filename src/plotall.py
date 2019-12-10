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
def plot_sir_all(resdir, outdir, nseeds=3):

    resdir = resdir[0]
    expspath = pjoin(resdir, 'exps.csv')
    df = pd.read_csv(expspath, index_col='expidx')

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

        maxI = np.max(aux.I)
        argmaxI = np.where(aux.I == np.amax(aux.I))

        title = 'topol:{},avgdegree:{},latt-thorus:{},\nbeta:{},gamma:{},gradspread:{}'. \
            format(v.topologymodel, v.avgdegree, v.lathoroidal, v.beta, v.gamma,
                   v.gaussianstd)
        ax[row, col].set_title(title, fontsize='small')
    plt.savefig(pjoin(outdir, 'sir_all.pdf'))

##########################################################
def plot_measures_luc(resdir, outdir, nseeds=3):
    resdir = resdir[0]
    expspath = pjoin(resdir, 'exps.csv')
    df = pd.read_csv(expspath, index_col='expidx')

    fig, ax = plt.subplots(1, 6, figsize=(30, 5))

    mycols = list(df.columns)
    mycols.remove('randomseed')
    mycols.remove('hostname')
    df_sorted = df.sort_values(mycols)
    plt.tight_layout(pad=2.5, h_pad=3.0, w_pad=0.5)

    nrows = df_sorted.shape[0]
    IminusS = np.ndarray(nrows, dtype=int)
    IminusR = np.ndarray(nrows, dtype=int)
    maxI = np.ndarray(nrows, dtype=int)
    argmaxI = np.ndarray(nrows, dtype=int)
    maxdiffI = np.ndarray(nrows, dtype=int)
    argmaxdiffI = np.ndarray(nrows, dtype=int)

    for j, expidx in enumerate(df_sorted.index):
        if not os.path.isdir(pjoin(resdir, expidx)): continue
        summarypath = pjoin(resdir, expidx, 'sir.csv')
        aux = pd.read_csv(summarypath)

        aux2 = aux.I - aux.S
        aux2 = aux2[aux2 > 0]
        IminusS[j] = np.sum(aux2)

        aux2 = aux.I - aux.R
        aux2 = aux2[aux2 > 0]
        IminusR[j] = np.sum(aux2)

        maxI[j] = np.max(aux.I)
        argmaxI[j] = np.where(aux.I == maxI[j])[0][0]

        aux2 = aux.I[1:].to_numpy()-aux.I[:-1].to_numpy()
        maxdiffI[j] = np.max(aux2)
        argmaxdiffI[j] = np.where(aux2 == maxdiffI[j])[0][0]

    ax[0].hist(IminusS)
    ax[0].set_title('I-S (positive)')
    ax[1].hist(IminusR)
    ax[1].set_title('I-R (positive)')
    ax[2].hist(maxI)
    ax[2].set_title('Max I')
    ax[3].hist(argmaxI)
    ax[3].set_title('Mode I')
    ax[4].hist(maxdiffI)
    ax[4].set_title('Maximum diff I')
    ax[5].hist(argmaxdiffI)
    ax[5].set_title('Mode of diff I')

    plt.savefig(pjoin(outdir, 'measures_luc.pdf'))

##########################################################
def plot_pca(resdir, outdir):
    expspath = pjoin(resdir[0], 'exps.csv')
    df = pd.read_csv(expspath, index_col='expidx')

    # import matplotlib.pyplot as plt
    # plotrows = int(df.shape[0]/nseeds)
    # fig, ax = plt.subplots(plotrows, nseeds, figsize=(8, plotrows*2))

    mycols = list(df.columns)
    # mycols.remove('randomseed')
    # mycols.remove('hostname')
    # df_sorted = df.sort_values(mycols)
    # df[df['lathoroidal'] == 'True']['lathoroidal'] = 1
    # input(df[df['lathoroidal'] == 'True'])
    # input(df.lathoroidal)
    # df[df['lathoroidal'] == 'False']['lathoroidal'] = 0
    # input(df.lathoroidal)
    # df[df['lathoroidal'] == '-1']['lathoroidal'] = -1
    # input(df.lathoroidal)
    # input(type(df.lathoroidal))

    df_sorted = df.sort_values(['topologymodel', 'beta', 'gamma', 'gaussianstd', 'avgdegree'])
    # plt.tight_layout(pad=2.5, h_pad=3.0, w_pad=0.5)
    # input(np.unique(df_sorted.gaussianstd))
    # print('Filtering by avgdegree=16')
    # df_sorted = df_sorted[df_sorted.avgdegree == 16]
    print('Filtering by seed=0')
    df_sorted = df_sorted[df_sorted.randomseed == 0]
    print('Filtering by thoroidal=-1 or thoroidal=0')
    df_sorted = df_sorted[(df_sorted.lathoroidal == -1) | (df_sorted.lathoroidal == 0)]
    print('Filtering by gamma=0.75')
    df_sorted = df_sorted[(df_sorted.gamma == 0.75)]

    for j, expidx in enumerate(df_sorted.index):
        if not os.path.isdir(pjoin(resdir[0], expidx)): continue
        summarypath = pjoin(resdir[0], expidx, 'transmcount.csv')
        aux = pd.read_csv(summarypath)
        nrows = aux.shape[0]

        ERR_VAL = -1
        i_s = aux.I - aux.S
        if np.where(i_s > 0)[0].size == 0: i_equal_s = ERR_VAL
        else: i_equal_s = np.where(i_s > 0)[0][0]

        r_s = aux.R - aux.S
        if np.where(r_s > 0)[0].size == 0: r_equal_s = ERR_VAL
        else: r_equal_s = np.where(r_s > 0)[0][0]

        r_i = aux.R - aux.I
        if np.where(r_i > 0)[0].size == 0: r_equal_i = ERR_VAL
        else: r_equal_i = np.where(r_i > 0)[0][0]

        imax = np.max(aux.I)
        # imode = np.argmax(aux.I)
        imode = aux.I.idxmax()

        df_sorted.loc[expidx, 't'] = nrows
        df_sorted.loc[expidx, 'iequals'] = i_equal_s
        df_sorted.loc[expidx, 'requals'] = r_equal_s
        df_sorted.loc[expidx, 'requali'] = r_equal_i
        df_sorted.loc[expidx, 'imax'] = imax
        df_sorted.loc[expidx, 'imode'] = imode

    X_orig = df_sorted[['topologymodel', 'beta', 'gamma', 'gaussianstd', 'avgdegree', 't', 'iequals', 'requals', 'requali', 'imax', 'imode']]
    print(X_orig)
    import copy
    X = copy.copy(X_orig)
    X['topologymodel'] = X_orig.topologymodel.map({'la':0, 'er':1, 'ba':2, 'ws':3})
    X = StandardScaler().fit_transform(X.astype(float))
    n, m = X.shape
    V = np.dot(X.T, X) / (n-1)
    values, vectors = np.linalg.eig(V) #non necessarily sorted
    P = np.dot(X, vectors)
    inds = np.argsort(values)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit(X)
    Y = pca.transform(X)
    print(pca.explained_variance_)
    print(pca.explained_variance_ratio_)

    ##########################################################
    fig, ax = plt.subplots(4,2, figsize=(20,40))
    ax[0][0].scatter(Y[:, 0], Y[:, 1])
    
    for i, y in enumerate(Y):
        x = X_orig.iloc[i]
        txt = '{},{},{},{},{},{}'.format(x.topologymodel, x.gaussianstd,
                int(x.t), int(x.iequals), int(x.requals), int(x.requali))
        ax[0][0].annotate(txt, (y[0]+np.random.rand()*0.1, y[1]+np.random.rand()*0.1))
    ax[0][0].set_title('topology, gaussianstd, T_converg, T(I=S), T(R=S), T(R=I)')

    ##########################################################
    mycmap = 'cividis'
    myedgecolor = None

    ##########################################################
    # fig, ax = plt.subplots(1, figsize=(10,10))
    c = X_orig.topologymodel.map({'la':0, 'er':1, 'ba':2}).values

    ax[0][1].scatter(Y[X_orig.topologymodel == 'la'][:, 0], Y[X_orig.topologymodel == 'la'][:, 1], c='blue', label='Lattice')
    ax[0][1].scatter(Y[X_orig.topologymodel == 'er'][:, 0], Y[X_orig.topologymodel == 'er'][:, 1], c='orange', label='ER')
    ax[0][1].scatter(Y[X_orig.topologymodel == 'ba'][:, 0], Y[X_orig.topologymodel == 'ba'][:, 1], c='green', label='BA')
    
    ax[0][1].set_title('PCA colored by topology')
    ax[0][1].legend(title='Topology')

    ########################################################## gaussian std
    aux = ax[1][0].scatter(Y[:, 0], Y[:, 1], c=X_orig.gaussianstd, cmap=mycmap, edgecolor=myedgecolor)
    
    # ax[1][1].set_title('PCA colored by T such that I=S')
    ax[1][0].set_title('PCA colored by gaussian std')
    fig.colorbar(aux, ax=ax[1][0])
    # ax[1][0].legend(title='Gaussian std')

    ########################################################## i equal s
    aux = ax[1][1].scatter(Y[:, 0], Y[:, 1], c=X_orig.iequals, cmap=mycmap, edgecolor=myedgecolor)
    
    ax[1][1].set_title('PCA colored by T such that I=S')
    fig.colorbar(aux, ax=ax[1][1])

    ########################################################## r equal s
    aux = ax[2][0].scatter(Y[:, 0], Y[:, 1], c=X_orig.requals, cmap=mycmap, edgecolor=myedgecolor)
    
    ax[2][0].set_title('PCA colored by T such that R=S')
    fig.colorbar(aux, ax=ax[2][0])

    ########################################################## r equal i
    aux = ax[2][1].scatter(Y[:, 0], Y[:, 1], c=X_orig.requali, cmap=mycmap, edgecolor=myedgecolor)
    
    ax[2][1].set_title('PCA colored by T such that R=I')
    fig.colorbar(aux, ax=ax[2][1])

    ########################################################## I max
    aux = ax[3][0].scatter(Y[:, 0], Y[:, 1], c=X_orig.imax, cmap=mycmap, edgecolor=myedgecolor)
    ax[3][0].set_title('PCA colored by maximum value of I')
    fig.colorbar(aux, ax=ax[3][0])

    ########################################################## I mode
    aux = ax[3][1].scatter(Y[:, 0], Y[:, 1], c=X_orig.imode, cmap=mycmap, edgecolor=myedgecolor)
    ax[3][1].set_title('PCA colored by T when for maxI')
    fig.colorbar(aux, ax=ax[3][1])

    ##########################################################
    plt.savefig('/tmp/plots.pdf')

##########################################################
def plot_francisco(resdir, outdir):
    expspath = pjoin(resdir[0], 'exps.csv')
    df = pd.read_csv(expspath, index_col='expidx')
    df_sorted = df.sort_values(['topologymodel', 'beta', 'gamma', 'gaussianstd', 'avgdegree'])
    print('Filtering by seed=0')
    df_sorted = df_sorted[df_sorted.randomseed == 0]
    print('n:{}'.format(df_sorted.shape[0]))

    print('Filtering by thoroidal=-1 or thoroidal=0')
    df_sorted = df_sorted[(df_sorted.lathoroidal == -1) | (df_sorted.lathoroidal == 0)]
    print('n:{}'.format(df_sorted.shape[0]))

    print('Filtering by gamma=0.75')
    df_sorted = df_sorted[(df_sorted.gamma == 0.75)]
    print('n:{}'.format(df_sorted.shape[0]))

    print('Filtering by avgdegree=16')
    df_sorted = df_sorted[(df_sorted.avgdegree == 16) | (df_sorted.topologymodel == 'la')]
    print('n:{}'.format(df_sorted.shape[0]))

    print('Filtering by mobilityratio=.5')
    df_sorted = df_sorted[(df_sorted.mobilityratio == 0.5)]
    print('n:{}'.format(df_sorted.shape[0]))

    # for std_ in np.unique(df_sorted.gaussianstd):
        # print(std_)
    tops = np.unique(df_sorted.topologymodel)
    # threshepoch = 700
    for threshepoch in [300, 500, 700]:
        # stds = np.unique(df_sorted.gaussianstd)
        # print(stds)
        fig, ax = plt.subplots(1, 4, figsize=(24, 6))
        charti = 0
        for top in tops:
            aux = df_sorted[df_sorted.topologymodel == top]

            valid = [] # filtering incomplete execution
            recoveredratios = []
            for j, expidx in enumerate(aux.index):
                if not os.path.exists(pjoin(resdir[0], expidx, 'transmcount.csv')):
                    continue
                valid.append(expidx)
                summarypath = pjoin(resdir[0], expidx, 'transmcount.csv')
                aux2 = pd.read_csv(summarypath)

                n = int(aux2.iloc[-1].S + aux2.iloc[-1].I + aux2.iloc[-1].R )
                if aux2.shape[0] <= threshepoch:
                    r = aux2.iloc[-1].R/n
                else:
                    r = aux2.iloc[threshepoch].R/n
                recoveredratios.append(r)

            aux = aux.loc[valid]
            aux['recoveredratio'] = recoveredratios

            # Generating colormap
            categories = np.unique(aux.gaussianstd)
            colors = np.linspace(0, 1, len(categories))
            colordict = dict(zip(categories, colors))
            mycmap = aux.gaussianstd.apply(lambda x: colordict[x])

            colors_ = ['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0','#f0027f','#bf5b17','#666666']
            for ii, std_ in enumerate(np.unique(aux.gaussianstd)):
                ax[charti].plot(aux[aux.gaussianstd==std_].beta,
                           aux[aux.gaussianstd==std_].recoveredratio,
                           marker='o', c=colors_[ii], label=str(std_))

            ax[charti].legend(title='Gaussian std')
            ax[charti].set_title('Topology {}'.format(top))
            ax[charti].set_xlabel('Beta')
            ax[charti].set_xlim(left=0, right=1)
            ax[charti].set_ylim(bottom=0, top=1)
            ax[charti].set_ylabel('Recovered ratio')
            charti += 1
        plt.savefig(pjoin(outdir, '{}.pdf'.format(threshepoch)))
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
    plot_francisco(args.resdir, outdir)

if __name__ == "__main__":
    main()

