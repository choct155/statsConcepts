import numpy as np
import pytest
from statsconcepts.Likelihood import Gaussian
from typing import Iterable, Callable, Tuple

# @pytest.fixture(scope="module")
# def standard(mu: float = 0, sigma: float = 1) -> Gaussian:
#     return Gaussian(mu, sigma)
# 
# @pytest.fixture
# def translate(mu: float = 3, sigma: float = 1) -> Gaussian:
#     return Gaussian(mu, sigma)
# 
# @pytest.fixture
# def wide(mu: float = 1, sigma: float = 5) -> Gaussian:
#     return Gaussian(mu, sigma)

class TestGaussian:

    standard = Gaussian(0, 1)
    translate = Gaussian(3, 1)
    wide = Gaussian(1, 5)

    def likelihood_calc(self, ll: Callable[[]]

    def test_likelihood_compare(self) -> None:
        """
        Standard should fit the best, followed by wide. Translate should be the worst
        """
        sample = np.random.normal(0, 1, size=100)
        def ll(g: Gaussian, xs: Iterable[float]) -> float:
            return g.likelihood(xs, log=False)
        lls = [
            ("standard", ll(self.standard, sample)),
            ("translate", ll(self.translate, sample)),
            ("wide", ll(self.wide, sample))
        ]
        lls.sort(key=lambda dist_run: dist_run[1])
        assert ["standard", "translate", "wide"] == [item[0] for item in lls]

    def test_log_likelihood_compare(self) -> None:
        """
        Standard should fit the best, followed by wide. Translate should be the worst
        """
        sample = np.random.normal(0, 1, size=100)
        def ll(g: Gaussian, xs: Iterable[float]) -> float:
            return g.likelihood(xs, log=True)
        lls = [
            ("standard", ll(self.standard, sample)),
            ("translate", ll(self.translate, sample)),
            ("wide", ll(self.wide, sample))
        ]
        lls.sort(key=lambda dist_run: dist_run[1])
        assert ["standard", "translate", "wide"] == [item[0] for item in lls]

    def test_likelihood_transform(self) -> None:
        """
        The likelihood rankings should not be altered by the log transform.
        """
