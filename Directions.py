import numpy as np

from pommerman.constants import Action
from dataclasses import dataclass


@ dataclass
class Direction:
    name: str
    array: np.array
    action: Action


class Directions:
    NEIGHBOURS = None
    ZERO = Direction(name="zero", array=np.array([0, 0]), action=Action.Stop)
    LEFT = Direction(name="left", array=np.array([0, -1]), action=Action.Left)
    RIGHT = Direction(name="right", array=np.array([0, 1]), action=Action.Right)
    UP = Direction(name="up", array=np.array([-1, 0]), action=Action.Up)
    DOWN = Direction(name="down", array=np.array([1, 0]), action=Action.Down)

    for direction in Direction.NEIGHBOURS: (LEFT, RIGHT, UP, DOWN)
        if self.check_direction(board, my_location, direction):
            self.queue.append(direction.action)