import numpy as np
import matplotlib as plt

class Point:

    def __init__(self, d1: float, d2: float):
        self.d1 = d1
        self.d2 = d2

    def __repr__(self) -> str:
        return "Point({}, {})".format(self.d1, self.d2)

    def __eq__(self, that_point: "Point") -> bool:
        d1_eq = self.d1 == that_point.d1
        d2_eq = self.d2 == that_point.d2
        return d1_eq & d2_eq

    def distance_to(self, that_point: "Point") -> float:
        d1_dist_sq = (that_point.d1 - self.d1)**2
        d2_dist_sq = (that_point.d2 - self.d2)**2
        return np.sqrt(d1_dist_sq + d2_dist_sq)



class Vector:

    def __init__(self, initial_pt: Point, terminal_pt: Point):
        self.initial_pt = initial_pt
        self.terminal_pt = terminal_pt

    def __repr__(self) -> str:
        return "Vector(initial: {}, terminal: {})".format(self.initial_pt, self.terminal_pt)

    def transform(self, matrix: np.array) -> "Vector":
        old_initial = np.array([self.initial_pt.d1, self.initial_pt.d2])
        old_terminal = np.array([self.terminal_pt.d1, self.terminal_pt.d2])
        new_initial = matrix.dot(old_initial)
        new_terminal = matrix.dot(old_terminal)
        out_initial = Point(new_initial[0], new_initial[1])
        out_terminal = Point(new_terminal[0], new_terminal[1])
        return Vector(out_initial, out_terminal)

    def plot(self, ax: plt.axis, **kwargs):
        x = [self.initial_pt.d1, self.terminal_pt.d1]
        y = [self.initial_pt.d2, self.terminal_pt.d2]
        ax.plot(x, y, **kwargs)


    def plot_arrow(self, ax: plt.axis, head_width=0.1, **kwargs):
        x = self.initial_pt.d1
        y = self.initial_pt.d2
        dx = self.terminal_pt.d1 - self.initial_pt.d1
        dy = self.terminal_pt.d2 - self.initial_pt.d2
        return ax.arrow(
            x,
            y,
            dx,
            dy,
            length_includes_head=True,
            head_width = head_width,
            **kwargs
        )

