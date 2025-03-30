import numpy as np
import pymc as pm
import matplotlib.pylab as plt
import seaborn as sb
import pandas as pd
from typing import Set, List, Dict, Callable, Tuple
import graphviz as gv


class ForkingTree:

    def __init__(self, n_w: int, n_l: int, n_draws: int) -> "ForkingTree":
        self.n_w: int = n_w
        self.n_l: int = n_l
        self.n: int = self.n_w + self.n_l
        self.n_draws: int = n_draws
        self.root_color: str = "#999897"
        self.w_color: str = "#557bc2"
        self.l_color: str = "#b8a688"
        self.g: gv.Graph = self.init_graph()
        

    def init_graph(self) -> gv.Graph:
        g: gv.Graph = gv.Graph("ForkingTree")
        g.node("0", "", shape="square", color=self.root_color, style="filled")
        g = self.next_draw(g_in=g, node="0", draws_remaining=self.n_draws)
        return g


    def next_draw(self, g_in: gv.Graph, node: str, draws_remaining: int) -> gv.Graph:
        if draws_remaining == 0:
            return g_in
        options: List[Tuple[int,str]] = [(i, node+str(i)) for i in range(self.n)]
        for option in options:
            if option[0] < self.n_w:
                g_in.edge(node, option[1], color=self.root_color)
                g_in.node(
                    option[1], 
                    color="red",
                    fillcolor=self.w_color, 
                    style="filled", 
                    shape="circle",
                    label="",
                    #edgecolor="red",
                    width="0.2"
                )
                self.next_draw(g_in=g_in, node=option[1], draws_remaining=draws_remaining - 1)
            else:
                g_in.edge(node, option[1], color=self.root_color)
                g_in.node(
                    option[1], 
                    color=self.l_color, 
                    style="filled", 
                    shape="circle",
                    label="",
                    width="0.2"
                )
                self.next_draw(g_in=g_in, node=option[1], draws_remaining=draws_remaining - 1)
    
        return g_in