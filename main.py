import seaborn as sb
# from matplotlib.figure import Figure
# from matplotlib.axes import Axes
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from typing import List, Dict
# import pandas as pd
import xarray as xr
from statsconcepts.regression import OLS, RegModel

def run() -> None:
    ds: xr.Dataset = xr.Dataset(sb.load_dataset("mpg"))
    ds["demeaned_mpg"] = ds["mpg"] - ds["mpg"].mean()
    ds["demeaned_weight"] = ds["weight"] - ds["weight"].mean()
    analytic: RegModel = OLS.fit("demeaned_mpg ~ demeaned_weight", ds)
    print(analytic)

    y: tf.Tensor = tf.constant(ds["demeaned_mpg"].data)
    x: tf.Tensor = tf.concat([
        tf.ones((len(y), 1), dtype=tf.float64),
        tf.reshape(ds["demeaned_weight"].data, shape=(len(y), 1))
    ], axis=1)

    def bayes_likelihood(beta0: tf.Tensor, beta1: tf.Tensor, sigma: tf.Tensor) -> tfd.Independent:
        beta = tf.concat([
            tf.reshape(beta0, shape=(1, 1)),
            tf.reshape(beta1, shape=(1, 1))
        ], axis=0)
        out: tfd.Independent = tfd.Independent(tfd.Normal(
            loc=tf.matmul(x, beta),
            scale=sigma
        ), reinterpreted_batch_ndims=1)
        return out


    model_ols: tfd.JointDistributionNamed = tfd.JointDistributionNamed(dict(
        beta0 = tfd.Normal(loc=tf.cast(0., dtype=tf.float64), scale=1.),
        beta1 = tfd.Normal(loc=tf.cast(0., dtype=tf.float64), scale=0.01),
        sigma = tfd.Uniform(low=tf.cast(7., dtype=tf.float64), high=8.),
        likelihood = bayes_likelihood
    ))

    def target_log_prob_fn(beta0: float, beta1: float, sigma: float) -> tf.Tensor:
        params: Dict[str, float] = dict(
            beta0=beta0,
            beta1=beta1,
            sigma=sigma,
            y=y[0]
        )
        out: tf.Tensor = model_ols.log_prob(params)
        return out

    # target_log_prob_fn(0., -0.008, 7.86)
    num_results: int = int(1e4)
    num_burn_in_steps: int = int(1e3)

    hmc_kernel: tfp.mcmc.HamiltonianMonteCarlo = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=1.0,
        num_leapfrog_steps=3
    )

    adaptive_hmc_kernel: tfp.mcmc.SimpleStepSizeAdaptation = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=hmc_kernel,
        num_adaptation_steps=int(num_burn_in_steps * 0.8)
    )

    @tf.function
    def run_chain():

        samples, is_accepted = tfp.mcmc.sample_chain(
            num_results = num_results,
            current_state=[
                tf.constant([0.], dtype=tf.float64),
                tf.constant([1.], dtype=tf.float64),
                tf.constant([7.], dtype=tf.float64),
            ],
            num_burnin_steps=num_burn_in_steps,
            kernel=adaptive_hmc_kernel,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted
        )
        return samples

    run_chain()

if __name__ == '__main__':
    run()