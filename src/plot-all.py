#!/usr/bin/env python3
"""Plot the results from lattice-sir.py
"""

import argparse
import logging
from os.path import join as pjoin
from logging import debug, info

import datetime
import numpy as np
import pandas as pd
import scipy
import scipy.stats

import plotly.graph_objects as go
from ipywidgets import widgets

def plot_all(indir):
    """Plot results from lattice-sir.py

    Args:
    indir(str): path to the parent directory of the results
    """
    df = pd.read_csv(pjoin(indir, 'exps.csv'))
    modes = []
    argmins = []
    for idx in df.idx:
        aux = pd.read_csv(pjoin(indir, idx, 'sir.csv'))

        mymode = np.argmax(aux.I)
        modes.append(int(mymode))

        myargmin = np.argmin(aux.S)
        argmins.append(int(myargmin))

    df['mode_i'] = modes
    df['argmin_s'] = argmins
    gradstds = df['gradstd'].unique()

    modei = np.ndarray((len(gradstds), 2))
    argmins = np.ndarray((len(gradstds), 2))

    for i, g in enumerate(gradstds):
        rows = df.loc[df['gradstd'] == g]

        mymean = np.mean(rows['mode_i'])
        mystd = np.std(rows['mode_i'])
        modei[i][0] = mymean
        modei[i][1] = mystd

        mymean = np.mean(rows['argmin_s'])
        mystd = np.std(rows['argmin_s'])
        argmins[i][0] = mymean
        argmins[i][1] = mystd

    datamode = go.Scatter(
        x=gradstds,
        y=modei[:, 0],
        name='Arg max of I',
        line=dict(width=4),
        error_y=dict(
            type='data', # value of error bar given in data coordinates
            array=modei[:, 1],
            visible=True)
    )
    dataargmin = go.Scatter(
        x=gradstds,
        y=argmins[:, 0],
        line=dict(width=4),
        name='Arg min of S',
        error_y=dict(
            type='data', # value of error bar given in data coordinates
            array=argmins[:, 1],
            visible=True)
    )
    plotdata = [datamode, dataargmin]
    plotlayout = go.Layout(
        title='Transmission time according to the uniformity',
        xaxis=dict(
            title='Standard deviation of the gaussian of the distribution of gradients'
        ),
        yaxis=dict(
            title='Time'
        )
    )
    fig = go.Figure(
        data=plotdata,
        layout=plotlayout
    )
    fig.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outdir', help='Directory containing the output from lattice-sir')
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s',
                        datefmt='%Y%m%d %H:%M', level=logging.DEBUG)
    plot_all(args.outdir)

if __name__ == "__main__":
    main()

