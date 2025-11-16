import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import seaborn.objects as so
import xarray as xr
import pandas as pd
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass
import networkx as nx


class DiscreteRandomVariable:
    def __init__(self, rv: Dict[int, float]) -> None:
        self.rv = rv

    def indicator_f(self) -> Callable[[float], Optional[int]]:
        print(self.rv)
        cum_rv: Dict[int, float] = dict(zip(self.rv.keys(), np.cumsum(list(self.rv.values()))))

        def ind(u: float) -> Optional[int]:
            for k, v in cum_rv.items():
                if u <= v:
                    return k
        return ind

    def sample(self, n: int) -> List[int]:
        ind_f: Callable[[float], Optional[int]] = self.indicator_f()
        s: List[Optional[int]] = [ind_f(np.random.uniform(0, 1)) for _ in range(n)]
        return [x for x in s if x is not None]


class InitialDistribution:

    def __init__(self, drv: DiscreteRandomVariable, states: Dict[str, int], n_draws: int = int(1e5)) -> None:
        self.drv: DiscreteRandomVariable = drv
        self.states: Dict[str, int] = states
        self.n_draws: int = n_draws
        self.samples: List[int] = self.sample()

    def sample(self) -> List[int]:
        return self.drv.sample(self.n_draws)

    def to_array(self) -> xr.DataArray:
        states, counts = np.unique(self.samples, return_counts=True)
        count_dict: Dict[str, int] = dict(zip(states, counts))

        for state in self.states.keys():
            if state not in count_dict:
                count_dict[state] = 0
        
        arr: xr.DataArray = xr.DataArray(
            np.array(list(count_dict.values())).reshape(len(self.states.keys()), 1),
            coords=[list(count_dict.keys()), [0]],
            dims=["from_state", "time"]
        )
        return arr

    def draw_hist(self) -> None:
        sb.histplot(self.samples, discrete=True)
        plt.xlabel("State")
        plt.ylabel("Count")
        plt.title("Initial Distribution Histogram")
        plt.show()

@dataclass
class TransitionProbability:
    from_state: str
    to_state: str
    probability: float

    @staticmethod
    def dict_to_list(transition_prob: Dict[str, Dict[str, float]]) -> List['TransitionProbability']:
        tp_list: List[TransitionProbability] = []
        for from_state, to_probs in transition_prob.items():
            for to_state, prob in to_probs.items():
                tp_list.append(TransitionProbability(from_state, to_state, prob))
        return tp_list

    @staticmethod
    def to_array(tps: List['TransitionProbability'], states: Dict[str, int]) -> xr.DataArray:
        n_states: int = len(states)
        matrix: np.ndarray = np.zeros((n_states, n_states))
        for tp in tps:
            i: int = states[tp.from_state]
            j: int = states[tp.to_state]
            matrix[i, j] = tp.probability
        return xr.DataArray(matrix, coords=[list(states.keys()), list(states.keys())], dims=["from_state", "to_state"])

    @staticmethod
    def graph(tps: List['TransitionProbability']) -> nx.DiGraph:
        G: nx.DiGraph = nx.DiGraph()
        for tp in tps:
            G.add_edge(tp.from_state, tp.to_state, weight=tp.probability, data="weight")
        return G

    @staticmethod
    def draw_graph(tps: List['TransitionProbability']) -> None:
        G: nx.DiGraph = TransitionProbability.graph(tps)
        pos: Dict[str, np.ndarray] = nx.spring_layout(G)
        edge_labels: Dict[tuple, float] = {(tp.from_state, tp.to_state): tp.probability for tp in tps}
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.title("Transition Probability Graph")
        plt.show()


class MarkovChain:

    def __init__(self, initial_dist: InitialDistribution, transition_probabilities: List[TransitionProbability]) -> None:
        self.initial_dist: InitialDistribution = initial_dist
        self.transition_probabilities: List[TransitionProbability] = transition_probabilities

    @staticmethod
    def time_step(
        current_state: xr.DataArray, 
        transition_matrix: xr.DataArray,
        inflow: Optional[xr.DataArray] = None,
    ) -> xr.DataArray:
        current_time: int = current_state.coords["time"][-1]
        next_state_dist: np.ndarray = xr.dot(transition_matrix.T, current_state.sel(time=current_time)).values
        next_state: xr.DataArray = xr.DataArray(
            next_state_dist.reshape(len(next_state_dist), 1),
            coords=[current_state.coords["from_state"], [current_time + 1]],
            dims=current_state.dims
        )
        if inflow is not None:
            next_state += inflow.assign_coords(time=[current_time + 1])
        out: xr.DataArray = xr.concat([current_state, next_state], dim="time")
        return out

    def simulate(
        self, 
        n_steps: int, 
        inflow: Optional[xr.DataArray] = None # For now assume constant inflow
    ) -> xr.DataArray:
        initial_array: xr.DataArray = self.initial_dist.to_array()
        transition_matrix: xr.DataArray = TransitionProbability.to_array(
            tps=self.transition_probabilities,
            states=self.initial_dist.states
        )
        current_state: xr.DataArray = initial_array
        for _ in range(n_steps):
            current_state = MarkovChain.time_step(current_state, transition_matrix, inflow)
        return current_state
    
    @staticmethod
    def draw_state_composition(sim_data: xr.DataArray, states: List[str]) -> None:
        sim_df: pd.DataFrame = sim_data.transpose("time", "from_state").to_pandas()[states]
        plot_df: pd.DataFrame = sim_df.reset_index().melt(id_vars=["time"], value_vars=states, var_name="from_state", value_name="count")
        fig, ax = plt.subplots(figsize=(10, 6))
        sim_df.plot(kind="bar", stacked=True, ax=ax)