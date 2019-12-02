import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Iterable

class Distribution(metaclass=ABCMeta):

    # self.gaussian = lambda mu, sigma: 

    def __init__(self):
        self.distribution = distribution

    @abstractmethod
    def prob(self, x: float) -> float:
        pass

    @abstractmethod
    def log_prob(self, x) -> float:
        pass

    @abstractmethod
    def likelihood(self, xs: Iterable[float], log: bool = True) -> float:
        pass

    @abstractmethod
    def likelihood_ratio(self, xs: Iterable[float], otherDistribution: "Distribution", log: bool = True):
        pass

class Gaussian(Distribution):

    def __init__(self, mu: float, sigma: float) -> None:
        self.mu = mu
        self.sigma = sigma

    def prob(self, x: float) -> float:
        normExpr = (self.sigma * np.sqrt(2 * np.pi))**(-1)
        exponent = (-(x - self.mu) ** 2)/(2 * self.sigma ** 2)
        probX = normExpr * np.exp(exponent)
        return probX

    def log_prob(self, x) -> float:
        normExpr = (self.sigma * np.sqrt(2 * np.pi))**(-1)
        exponent = (-(x - self.sigma) ** 2)/(2 * self.sigma ** 2)
        logProbX = np.log(normExpr) - exponent
        return logProbX

    def likelihood(self, xs: Iterable[float], log: bool = True) -> float:
        if log == True:
            log_probs = np.array(list(map(self.log_prob, xs)))
            return log_probs.sum()
        else:
            probs = np.array(list(map(self.prob, xs)))
            return probs.prod()

    def likelihood_ratio(self, xs: Iterable[float], otherDistribution: "Distribution", log: bool = True):
        if log == True:
            thisLikelihood = self.likelihood(xs, True)
            thatLikelihood = otherDistribution.likelihood(xs, True)
            return thisLikelihood - thatLikelihood
        else:
            thisLikelihood = self.likelihood(xs, False)
            thatLikelihood = otherDistribution.likelihood(xs, False)
            return thisLikelihood / thatLikelihood


