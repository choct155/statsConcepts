from dataclasses import dataclass
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from typing import Union, ClassVar, Callable, Iterable, TypeVar, Tuple, List, Dict
import numpy as np
import xarray as xr
import seaborn as sb


A = TypeVar("A")


@dataclass
class SeriesPlotIn:
    label: str
    data: Iterable[float]
    color: str

@dataclass
class Viz:
    """
    Visualization class to add in the statistical comparison
    of subpopulations
    """
    data: xr.Dataset
    colors: ClassVar[sb.color_palette] = sb.color_palette("hls", 8)

    def univariate_hist(
            self,
            x: str,
            color: str,
            stat: Callable[[Iterable[float]], float] = np.mean,
            label: Callable[[str], str] = lambda stat_val: f"Avg = {stat_val}",
            figsize: Tuple[int, int] = (12, 5),
            title: str = "",
            **kwargs
    ) -> Axes:
        fig, ax = plt.subplots(figsize=figsize)
        series_input: SeriesPlotIn = SeriesPlotIn(x, self.data[x], color)
        out: Axes = Viz.dist_with_stat(ax, series_input, stat, vert=True, **kwargs)
        stat_val_desc: str = Viz.stat_label(self.data, x, stat, label)
        out.set_title(f"{title} ({stat_val_desc})")
        return out

    @staticmethod
    def univariate_compare(
            grp_1: SeriesPlotIn,
            grp_2: SeriesPlotIn,
            stat: Callable[[Iterable[float]], float] = np.mean,
            figsize: Tuple[int, int] = (12, 5),
            title: str = "",
            **kwargs
    ) -> Axes:
        fig, ax = plt.subplots(figsize=figsize)
        with_x: Axes = Viz.dist_with_stat(ax, grp_1, stat, vert=True, **kwargs)
        with_y: Axes = Viz.dist_with_stat(with_x, grp_2, stat, vert=True, **kwargs)
        stat_val_desc_1: str = Viz.stat_label(grp_1.data, stat)
        stat_val_desc_2: str = Viz.stat_label(grp_2.data, stat)
        with_y.set_title(f"{title} ({grp_1.label} {stat_val_desc_1}; {grp_2.label} {stat_val_desc_2})")
        return with_y

    @staticmethod
    def stat_label(data: Iterable[float], stat: Callable[[Iterable[float]], float] = np.mean) -> str:
        stat_val: float = stat(data)
        stat_val_desc: str = f"{stat.__name__}: {'{:.2f}'.format(stat_val)}"
        return stat_val_desc

    @staticmethod
    def despine(ax: Axes) -> Axes:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        return ax

    @staticmethod
    def dist(ax: Axes, series_input: SeriesPlotIn, **kwargs) -> Axes:

        if "edgecolor" not in kwargs.keys():
            kwargs.update({"edgecolor": "#ffffff"})

        out: Axes = sb.histplot(x=series_input.data, ax=ax, **kwargs)
        out = Viz.despine(out)
        for p in out.containers[-1].__dict__["patches"]:
            p.set_facecolor(series_input.color)
        return out

    @staticmethod
    def add_stat_line(
            ax: Axes,
            series_input: SeriesPlotIn,
            stat: Callable[[Iterable[float]], float] = np.mean,
            vert: bool = True,
            **kwargs
    ) -> Axes:
        
        if "linestyle" not in kwargs.keys():
            kwargs.update({"linestyle": "--"})
            
        stat_val: float = stat(series_input.data)
        
        if vert:
            ax.axvline(x=stat_val, color=series_input.color, **kwargs)
        else:
            ax.axhline(y=stat_val, color=series_input.color, **kwargs)

        return ax

    @staticmethod
    def dist_with_stat(
            ax: Axes,
            series_input: SeriesPlotIn,
            stat: Callable[[Iterable[float]], float] = np.mean,
            vert: bool = True,
            **kwargs
    ) -> Axes:
        with_dist: Axes = Viz.dist(ax, series_input, **kwargs)
        with_stat: Axes = Viz.add_stat_line(with_dist, series_input, stat, vert)
        return with_stat

    @staticmethod
    def multi_hist(
            ax: Axes,
            inputs: List[SeriesPlotIn],
            f: Callable[[Axes, SeriesPlotIn], Axes] = lambda ax, spi: Viz.dist_with_stat(ax, spi)
    ) -> Axes:
        h, *t = inputs
        updated: Axes = f(ax, h)
        for i in t:
            updated = f(updated, i)
        return updated

    @staticmethod
    def group_axes(
            grps: Iterable[A],
            n_cols: int,
            sharex: bool = False,
            sharey: bool = False,
            **kwargs
    ) -> Figure:
        grp_list: List[A] = list(grps)
        n_grps: int = len(grp_list)

        if n_grps % n_cols == 0:
            n_rows: int = n_grps // n_cols
        else:
            n_rows: int = (n_grps // n_cols) + 1

        fig: Figure = plt.Figure(**kwargs)
        for i in range(1, n_grps + 1):
            next_ax: Axes = fig.add_subplot(n_rows, n_cols, i)
            remove_yticks: bool = sharey and not (i % n_cols == 1)
            remove_xticks: bool = sharex and not (i > (n_grps - n_cols))
            if remove_yticks:
                next_ax.set_yticks([])
            if remove_xticks:
                next_ax.set_xticks([])
            
        return fig
