import plotly.graph_objects as go
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.framework.dtypes import DType
tfd = tfp.distributions
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List


class Noise:

    def __init__(
        self,
        dist: tfd.Distribution,
        params: Dict[str, tf.Tensor]
    ) -> None:
        self.dist: tfd.Distribution = dist
        self.params: Dict[str, tf.Tensor] = params
        self.rv: tfd.Distribution = self.dist(**self.params)

    def sample(self, n: int) -> tf.Tensor:
        return self.rv.sample(n)


class LinearTrend:
    def __init__(self, slopes: tf.Tensor, intercept: Optional[tf.Tensor] = None, dtype: DType = tf.float32) -> None:
        self.slopes: tf.Tensor = slopes
        self.intercept: tf.Tensor = tf.fill(self.slopes.shape, 0.) if intercept is None else intercept
        self.dtype: DType = dtype
        if self.slopes.shape != self.intercept.shape:
            raise ValueError("The center and slope vectors must have the same shape!")

    def support(self, n: int) -> tf.Tensor:
        base_idx: tf.Tensor = tf.cast(tf.range(n), dtype=self.dtype)
        origin_idx: tf.Tensor = SimUtils.tensor_repeat(base_idx, 1, self.slopes[0])
        idx: tf.Tensor = origin_idx + self.intercept
        return idx

    def sample(self, n: int) -> tf.Tensor:
        idx: tf.Tensor = self.support(n)
        sample: tf.Tensor = idx * tf.reshape(self.slopes, (1, len(self.slopes)))
        return sample

    def plot(self, n: int) -> go.Figure:
        support: tf.Tensor = self.support(n)
        sample: tf.Tensor = self.sample(n)
        if len(sample.shape) > 2:
            raise ValueError("Sorry! We can only plot in 3-D, which means 2-D arrays are the limit.")

        # if (len(sample.shape) == 1) or (sample.shape[1] == 1):
        #     plot_support: tf.Tensor = tf.reshape(support, [-1])
        #     plot_sample: tf.Tensor = tf.reshape(sample, [-1])
        #     fig: go.Figure = go.Figure()
        #     fig.add_trace(go.Scatter(
        #         x=plot_support,
        #         y=plot_sample,
        #         mode="lines"
        #     ))
        # else:




class SimUtils:

    @staticmethod
    def tensor_repeat(t: tf.Tensor, expand_dim: int, repeats: int) -> tf.Tensor:
        base: tf.Tensor = tf.expand_dims(t, expand_dim)
        bases: List[tf.Tensor] = [base for _ in range(repeats)]
        out: tf.Tensor = tf.concat(bases, axis=expand_dim)
        return out

class TensorViz:

    def __init__(self, data: tf.Tensor) -> None:
        self.data: tf.Tensor = data
        self.shape: Tuple[int, ...] = data.shape

    @staticmethod
    def single_ax_plot(
        data: tf.Tensor,
        title: str = "",
        labels: Optional[List[str]] = None,
        template: str = "plotly_white",
        **kwargs
    ) -> go.Figure:
        if len(data.shape) == 1:
            fig: go.Figure = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(data.shape[0])),
                y=data,
                mode="lines"
            ))
        else:
            fig: go.Figure = go.Figure()
            for i in range(data.shape[1]):
                fig.add_trace(go.Scatter(
                    x=list(range(data.shape[0])),
                    y=data[:, i],
                    mode="lines",
                    name=None if labels is None else labels[i]
                ))
        fig.update_layout(
            title=title,
            template=template,
            **kwargs
        )
        return fig

