from pommerman import characters
from pommerman.agents import BaseAgent
from pommerman.constants import Action, Item
from directions import Direction, Directions
import numpy as np


class MyAgent(BaseAgent):
    """ The version of base agent. """

    def __init__(self, character=characters.Bomber):
        super().__init__(character)
        self.queue = []


    def act(self, obs, action_space):
        # Main event that is being called on every turn.
        if not self.queue:
            my_location = obs['position']
            board = obs['board']
            goal_location = self.move_to_safe_location(obs, my_location)

            if self.can_place_bomb(obs["bomb_life"],obs["ammo"], my_location):
                self.queue.append(Action.Bomb)

            for direction in self.create_path(board, my_location, goal_location):
                self.queue.append(direction.action)

            # If cannot move in any direction, send a pass
            if not self.queue:
                self.queue.append(Action.Stop)

        return self.queue.pop(0)

    def create_path(self, board: np.array, my_location: tuple, goal_location: tuple) -> bool:
        """ BFS to a goal location.  Returns True if it can be reached."""
        to_visit = [my_location]
        visited = []

        came_from = dict()
        came_from[my_location] = None

        came_from_direction = dict()
        came_from_direction[my_location] = Directions.ZERO

        while to_visit:
            point = to_visit.pop(0)
            if point == goal_location: break
            for direction in Directions.NEIGHBOURS:
                # By making it a numpy can add the values
                new_point = tuple(np.array(point) + direction.array)

                # Either the row or column value is not on the board
                if not self.in_bounds(new_point):
                    continue

                # Has already visited that point
                if new_point in visited:
                    continue

                # Can it reach this point? -> Yes the add to visit list
                if self.check_direction(board, point, direction):
                    to_visit.append(new_point)
                    came_from[point] = new_point
                    came_from_direction[point] = direction

            visited.append(point)
        return self.reverse_path(came_from, came_from_direction, goal_location)

    def in_bounds(self, location: tuple) -> bool:
        """ Checks if a location is on the board, if it is returns True.  """
        return 0 <= min(location) and max(location) <= 7

    def check_direction(self, board: np.array, location: tuple, direction: Direction) -> bool:

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

    def reverse_path(self, came_from: dict, came_from_direction: dict, goal_location: tuple) -> list:
        """ Returns a list of directions from a start node with the parent None, leading to the goal node. """
        current = goal_location
        parent = came_from.get(goal_location, None)
        path = []

        while parent is not None:
            path.append(came_from_direction[current])
            current, parent = parent, came_from.get(parent, None)

            return list(reversed(path))

    def create_danger_map(self, obs: dict) -> np.ndarray:
        """
            Returns a map the size of the board, with the following positional encoding:

                0 : A safe place where you can move to.
                1 : A place that will kill you if you are there.
                >1 : A place that will be dangerous in the future (counts down to 1).
        """

        # Set our initial danger map
        danger_map = obs['flame_life']
        danger_map[danger_map > 0] = 1

        # Find all bomb locations, bomb timers and strength
        bombs = np.where(obs['bomb_life'] > 0)
        bombs_timers = map(int, obs['bomb_life'][bombs])
        bomb_strength = map(int, obs['bomb_blast_strength'][bombs])

        # Now we are going to set the danger information
        for row, col, timer, strength in zip(*bombs, bombs_timers, bomb_strength):

            # Reduce strength by one, since we are creating a `+` form, with the bomb as center.
            strength -= 1

            # Calculate the upper and lower ranges of the bombs (this is the + sign).
            row_low, row_high = max(row - strength, 0), min(row + strength, 7)
            col_low, col_high = max(col - strength, 0), min(col + strength, 7)

            # Set the information on the danger map, first row and then column.
            for row_danger in range(row_low, row_high + 1):
                danger_map[row_danger, col] = timer

            for col_danger in range(col_low, col_high + 1):
                danger_map[row, col_danger] = timer

        return danger_map

    def find_reachable_safe_location(self, board: np.ndarray, danger_map: np.ndarray, location: tuple) -> tuple:
        to_visit = [location]
        visited = []

        while to_visit:
            point = to_visit.pop(0)

            if danger_map[point] == 0:
                return point

            for direction in Directions.NEIGHBORS:
                new_point = tuple(np.array(point) + direction.array)

                # Filter out the bad points
                if not self.in_bounds(new_point) or new_point in visited or danger_map[new_point] > 0:
                    continue

                # If we can reach this point add the point to the to visit list
                if self.check_direction(board, point, direction):
                    to_visit.append(new_point)

            visited.append(point)

        # no safe place was found, so stay where you are and pray.
        return location

    def move_to_safe_location(self, obs, location: tuple):
        """ Returns a location to which we can safely move.  """

        # Create a mapping of positions and danger level
        danger_map = self.create_danger_map(obs)

        # Check if our current position is safe, if so we can go/stay there.
        return self.find_reachable_safe_location(obs['board'], danger_map, location)
