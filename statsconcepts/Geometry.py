import numpy as np
import matplotlib as plt
from typing import Sequence

class Point:

    def __init__(self, coords: Sequence[float]):
        self.coords = coords
        # Danger! This variable holds the state of the coordinate index
        self.coord_index = 0
        self.ndims = len(self.coords)

    def __repr__(self) -> str:
        m = map(str, self.coords)
        str_coords = ", ".join(m)
        return "Point({})".format(str_coords)

    def __eq__(self, that_point: "Point") -> bool:
        m = map(lambda this, that: this == that, self.coords, that_point.coords)
        return all(m)

    def __getitem__(self, idx: int):
        return self.coords[idx]

    def distance_to(self, that_point: "Point") -> float:
        """
        Calculates the l2 norm
        """
        m = map(lambda this, that: (that - this)**2, self.coords, that_point.coords)
        return np.sqrt(sum(m))

    def scale(self, const: float) -> "Point":
        m = map(lambda coord: coord * const, self.coords)
        return Point(list(m))

    def translate(self, delta: Sequence[float]) -> "Point":
        assert len(delta) == self.ndims, "Precisely one translation is required for each dimension"
        m = map(lambda coord, d: coord + d, self.coords, delta)
        return Point(list(m))



class Vector:
    """
    While it is true that numpy does have good handles for arrays (which we use), they generally
    do not have formulations for points, and certainly don't touch derivative concepts like divergence
    and curl. Building this here helps to map closely to material
    """

    def __init__(self, initial_pt: Point, terminal_pt: Point):
        self.initial_pt = initial_pt
        self.terminal_pt = terminal_pt
        self.length = initial_pt.distance_to(terminal_pt)

    def __repr__(self) -> str:
        return "Vector(initial: {}, terminal: {})".format(self.initial_pt, self.terminal_pt)

    def __eq__(self, that_vector: "Vector") -> bool:
        initial_eq = self.initial_pt == that_vector.initial_pt
        terminal_eq = self.terminal_pt == that_vector.terminal_pt
        return initial_eq & terminal_eq

    def add(self, that_vector: "Vector") -> "Vector":
        return Vector(self.initial_pt, that_vector.terminal_pt)

    def scale(self, const: float) -> "Vector":
        return Vector(self.initial_pt, self.terminal_pt.scale(const))

    def dot(self, that_vector: "Vector") -> float:
        # translate both terminal pts as if both vectors had the origin as the inital point
        this_terminal = self.terminal_pt.translate(self.initial_pt.scale(-1.))
        that_terminal = that_vector.terminal_pt.translate(that_vector.initial_pt.scale(-1.))
        # We can return the dot product from here
        m = map(lambda this, that: this * that, this_terminal, that_terminal)
        return sum(m)

    def cross_prod(self, that_vector: "Vector") -> "Vector":
        """
        While the cross product can be produced by way of capturing the determinants in each dimension
        we are going to
        Shit, for this to scale in dimension, perhaps the determinant method is unavoidable?
        """

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

