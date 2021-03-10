from pommerman import characters
from pommerman.agents import BaseAgent
from pommerman.constants import Action, Item
from serpentine.directions import Direction, Directions
import numpy as np


class MyAgent(BaseAgent):
    """ Our version of the base agent. """

    def __init__(self, character=characters.Bomber):
        super().__init__(character)
        self.queue = []

    def act(self, obs, action_space):
        # Main event that is being called on every turn.
        if not self.queue:
            my_location = obs['position']
            board = obs['board']

            if self.check_left(board, my_location):
                self.queue.append(Action.Left)
            """"Checks Left"""
            if self.check_right(board, my_location):
                self.queue.append(Action.Right)
            """Checks Right"""
            if self.check_up(board, my_location):
                self.queue.append(Action.Up)
            """Up"""
            if self.check_down(board, my_location):
                self.queue.append(Action.Down)
            """"Down"""
            # If cannot move in any direction, send a pass
            if not self.queue:
                self.queue.append(Action.Stop)

        goal_location = (2,2)

        if self.can_move_to (board, my_location, goal_location):
            print("Can now move to this location: ""goal_location")

        return self.queue.pop(0)


    def can_move_to(self, board: np.array, my_location: tuple, goal_location: tuple) -> bool:
        """ BFS to a goal location.  Returns True if it can be reached."""
        to_visit = [my_location]
        visited = []

        came_from = dict()
        came_from[my_location] = None

        came_from_direction = dict()
        came_from_direction[my_location] = Directions.ZERO

        while to_visit:
            point = to_visit.pop(0)

            for direction in Directions.NEIGHBOURS:
                #By making it a numpy we can add the values
                new_point = tuple(np.array(point) + direction.array)

                # Either the row or column value is not on the board
                if not self.in_bounds(new_point):
                    continue

                # Has already visited that point
                if new_point in visited:
                    continue

                if new_point == goal_location:
                    return True

                # Can it reach this point? -> Yes the add to visit list
                if self.check_direction_passable(board, point, direction):
                    to_visit.append(new_point)
                    came_from[point] = new_point
                    came_from_direction[point] = direction

            visited.append(point)
        return False

    def in_bounds(self, location: tuple) -> bool:
        """ Checks if a location is on the board, if it is returns True.  """
        return 0 <= min(location) and max(location) <= 7

    def check_direction(self, board: np.array, location: tuple, direction: Direction) -> bool:
       #pass
        """
            Checks for a given location and direction if the new location is a Passage.

            :param board: The game board
            :param location: The location from which you start to move
            :param direction: The direction in which you are going to move (This is the Direction class)
            :return: True if the new location is a passage, False otherwise.
        """

        new_location = np.array(location) + direction.array
        if not self.in_bounds(new_location):
            return False

        # Note that this is already a boolean (so no need for if statements)
        return board[tuple(new_location)] == Item.Passage.value