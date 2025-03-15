"""
For graphical representations of data.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from pyiron_workflow import as_function_node


@as_function_node("fig")
def PlotDataFrame(df: pd.DataFrame, x: list | np.ndarray):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    plot = df.plot(x=x, ax=ax)
    return plt.show()


@as_function_node("fig")
def PlotDataFrameXY(df: pd.DataFrame, x: list | np.ndarray):
    from matplotlib import pyplot as plt

    # Check if dataframe has only two columns and x parameter is not provided.
    if df.shape[1] == 2 and x is None:
        columns = df.columns
        x = columns[0]  # First column for x-axis.
        y = columns[1]  # Second column for y-axis.
        x_label, y_label = x, y
    else:
        x_label = x if isinstance(x, str) else "x label not defined"
        y_label = "y label not defined"

    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    df.plot(x=x, y=y, ax=ax)

    return plt.show()


@as_function_node("fig")
def Scatter(
    x: list | np.ndarray, y: list | np.ndarray
):
    from matplotlib import pyplot as plt

    plt.scatter(x, y)
    return plt.show()


@as_function_node("fig")
def ShowArray(mat: np.ndarray):
    from matplotlib import pyplot as plt

    plt.imshow(mat)
    return plt.show()


@as_function_node("fig")
def Histogram(x: list | np.ndarray, bins: int = 50):
    from matplotlib import pyplot as plt

    plt.hist(x, bins=bins)
    return plt.show()


@as_function_node("figure")
def Plot(
    y: list | np.ndarray | pd.core.series.Series,
    x: list | np.ndarray | pd.core.series.Series,
    axis: object,
    title: str = "",
    color: str = "b",
    symbol: str = "o",
    legend_label: str = "",
):
    from matplotlib import pyplot as plt

    # If x is not provided, generate a default sequence
    x = np.arange(len(y)) if x is None else x

    if axis is None:
        axis = plt
        axis.title = title
        axis.plot(x, y, color=color, marker=symbol, label=legend_label)
        figure = axis.show()

    else:
        axis.set_title(title)  # Set the title of the plot
        axis.plot(x, y, color=color, marker=symbol, label=legend_label)
        figure = axis

    return figure


@as_function_node("linspace")
def Linspace(
    start: int | float = 0.0,
    stop: int | float = 1.0,
    num: int = 50,
    endpoint: bool = True,
):
    from numpy import linspace

    return linspace(start, stop, num, endpoint=endpoint)
