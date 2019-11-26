import numpy as np

class PDF:

    # self.gaussian = lambda mu, sigma: 

    def __init__(self, distribution):
        self.distribution = distribution


class Gaussian:

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def prob(self, x):
        normExpr = (self.sigma * np.sqrt(2 * np.pi))**(-1)
        exponent = (-(x - self.mu) ** 2)/(2 * self.sigma ** 2)
        probX = normExpr * np.exp(exponent)
        return probX

    def logProb(self, x):
        normExpr = (self.sigma * np.sqrt(2 * np.pi))**(-1)
        exponent = (-(x - self.sigma) ** 2)/(2 * self.sigma ** 2)
        logProbX = np.log(normExpr) - exponent
        return logProbX

    def likelihood(self, xs, log = True):
        if log == True:
            logProbs = np.array(list(map(self.logProb, xs)))
            return logProbs.sum()
        else:
            probs = np.array(list(map(self.prob, xs)))
            return probs.prod()

    def likelihoodRatio(self, xs, otherGaussian, log = True):
        if log == True:
            thisLikelihood = self.likelihood(xs, True)
            thatLikelihood = otherGaussian.likelihood(xs, True)
            return thisLikelihood - thatLikelihood
        else:
            thisLikelihood = self.likelihood(xs, False)
            thatLikelihood = otherGaussian.likelihood(xs, False)
            return thisLikelihood / thatLikelihood


