from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sb
import pandas as pd
import xarray as xr
import tensorflow as tf
import tensorflow_probability as tfp
from dataclasses import dataclass
from patsy.design_info import DesignInfo, DesignMatrix
from patsy.highlevel import dmatrices
from typing import Optional, Callable
from IPython.display import display


@dataclass
class RegModel:
    data: xr.Dataset
    y: DesignMatrix
    x: DesignMatrix
    betas: tf.Tensor
    estimator_label: str

    def __post_init__(self) -> None:
        self._y_design_info: DesignInfo = self.y.design_info
        self._x_design_info: DesignInfo = self.x.design_info

    def __repr__(self):
        if not self.betas:
            display(self.data)
            print("No model has yet been estimated")

        display(self.data)
        summary = (f"{self.estimator_label}"
                   "  Model: %s ~ %s\n"
                   "  Regression (beta) coefficients:\n"
                   % (self._y_design_info.describe(),
                      self._x_design_info.describe()))
        for name, value in zip(self._x_design_info.column_names, self.betas):
            summary += "    %s:  %0.3g\n" % (name, value[0])
        return summary


@dataclass
class OLS:

    @staticmethod
    def fit(formula: str, data: xr.Dataset) -> RegModel:
        y: DesignMatrix
        x: DesignMatrix
        y, x = dmatrices(formula, data)
        y_tensor: tf.Tensor = tf.constant(y)
        x_tensor: tf.Tensor = tf.constant(x)
        betas: tf.Tensor = OLS.fit_exec(y_tensor, x_tensor)
        out: RegModel = RegModel(data, y, x, betas, "Direct Ordinary Least Squares")
        return out

    @staticmethod
    def fit_exec(y: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        cov: tf.Tensor = tf.linalg.matmul(tf.transpose(x), y)
        var_design: tf.Tensor = tf.linalg.matmul(tf.transpose(x), x)
        inv_var_design: tf.Tensor = tf.linalg.inv(var_design)
        betas: tf.Tensor = tf.matmul(inv_var_design, cov)
        return betas
