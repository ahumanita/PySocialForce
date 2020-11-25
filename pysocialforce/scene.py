"""This module tracks the state odf scene and scen elements like pedestrians, groups and obstacles"""
from typing import List

import numpy as np

from pysocialforce.utils import stateutils


class PedState:
    """Tracks the state of pedstrains and social groups"""

    def __init__(self, simulator, state, groups, border, config):
        self.default_tau = config("tau", 0.5)
        self.step_width = config("step_width", 0.4)
        self.agent_radius = config("agent_radius", 0.35)
        self.max_speed_multiplier = config("max_speed_multiplier", 1.3)
        self.simulator = simulator

        self.border = border

        self.max_speeds = None
        self.initial_speeds = None

        self.ped_states = []
        self.group_states = []

        self.update(state, groups)

    def update(self, state, groups):
        self.state = state
        self.groups = groups

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        tau = self.default_tau * np.ones(state.shape[0])
        if state.shape[1] < 7:
            escaped = np.zeros(state.shape[0])
            targeted_exit = -np.ones(state.shape[0])
            state = np.concatenate((state, np.expand_dims(tau, -1)), axis=-1)
            state = np.concatenate((state, np.expand_dims(escaped, -1)), axis=-1)
            self._state = np.concatenate((state, np.expand_dims(targeted_exit, -1)), axis=-1)
        else:
            self._state = state
        if self.initial_speeds is None:
            self.initial_speeds = self.speeds()
        self.max_speeds = self.max_speed_multiplier * self.initial_speeds
        self.ped_states.append(self._state.copy())

    def get_states(self):
        return np.stack(self.ped_states), self.group_states

    def size(self) -> int:
        return self.state.shape[0]

    def pos(self) -> np.ndarray:
        return self.state[:, 0:2]

    def vel(self) -> np.ndarray:
        return self.state[:, 2:4]

    def goal(self) -> np.ndarray:
        return self.state[:, 4:6]

    def set_goal(self, p, new_goal) :
        self.state[p, 4:6] = new_goal

    def set_exit_goal(self, p, exit_id) :
        self.set_goal(p, self.simulator.get_exits()[exit_id][:2])
        self.state[p,8] = exit_id

    def tau(self):
        return self.state[:, 6:7]

    def escaped(self) :
        return self.state[:,7]

    def targeted_exit(self) :
        return self.state[:,8]

    def get_nr_escaped(self) :
        return np.sum(self.state[:,7])
    
    def get_nr_peds(self) :
        return self.state.shape[0]

    def speeds(self):
        """Return the speeds corresponding to a given state."""
        return stateutils.speeds(self.state)

    def step(self, force, groups=None):
        """Move peds according to forces"""
        # desired velocity
        desired_velocity = self.vel() + self.step_width * force
        desired_velocity = self.capped_velocity(desired_velocity, self.max_speeds)
        # stop when arrived
        desired_velocity[stateutils.desired_directions(self.state)[1] < 0.5] = [0, 0]

        # update state
        next_state = self.state
        next_state[:, 0:2] += desired_velocity * self.step_width
        next_state[:, 2:4] = desired_velocity
        next_groups = self.groups
        if self.border is not None:
            escaped_rows = np.where((next_state[:,0] <= self.border[0]) | (next_state[:,0] >= self.border[1]) | (next_state[:,1] <= self.border[2]) | (next_state[:,1] >= self.border[3]))[0]
            if escaped_rows.size > 0 :
                next_state[escaped_rows,7] = 1
        if groups is not None:
            next_groups = groups
        self.update(next_state, next_groups)

    # def initial_speeds(self):
    #     return stateutils.speeds(self.ped_states[0])

    def desired_directions(self):
        return stateutils.desired_directions(self.state)[0]

    def set_exits_as_targets(self) :
        # TODO: efficient way to do this?
        exits = self.simulator.get_exits()
        if exits is not None :
            pos = self.pos()
            for p in range(self.get_nr_peds()) :
                closest = 0
                closest_dist = np.linalg.norm(pos[p]-exits[0][:2])
                for e in range(1,len(exits)) :
                    dist = np.linalg.norm(pos[p]-exits[e][:2])
                    if dist < closest_dist :
                        closest = e
                        closest_dist = dist
                self.set_exit_goal(p,closest)




    @staticmethod
    def capped_velocity(desired_velocity, max_velocity):
        """Scale down a desired velocity to its capped speed."""
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        factor = np.minimum(1.0, max_velocity / desired_speeds)
        factor[desired_speeds == 0] = 0.0
        return desired_velocity * np.expand_dims(factor, -1)

    @property
    def groups(self) -> List[List]:
        return self._groups

    @groups.setter
    def groups(self, groups: List[List]):
        if groups is None:
            self._groups = []
        else:
            self._groups = groups
        self.group_states.append(self._groups.copy())

    def has_group(self):
        return self.groups is not None

    # def get_group_by_idx(self, index: int) -> np.ndarray:
    #     return self.state[self.groups[index], :]

    def which_group(self, index: int) -> int:
        """find group index from ped index"""
        for i, group in enumerate(self.groups):
            if index in group:
                return i
        return -1

class EnvState:
    """State of the environment obstacles"""

    def __init__(self, obstacles, fires=None, exits=None, resolution=10):
        self.resolution = resolution
        self.obstacles = obstacles
        self.fires = fires
        self.exits = exits

    @property
    def obstacles(self) -> List[np.ndarray]:
        """obstacles is a list of np.ndarray"""
        return self._obstacles

    @obstacles.setter
    def obstacles(self, obstacles):
        """Input an list of (startx, endx, starty, endy) as start and end of a line"""
        if obstacles is None:
            self._obstacles = []
        else:
            self._obstacles = []
            for startx, endx, starty, endy in obstacles:
                samples = int(np.linalg.norm((startx - endx, starty - endy)) * self.resolution)
                line = np.array(
                    list(
                        zip(np.linspace(startx, endx, samples), np.linspace(starty, endy, samples))
                    )
                )
                self._obstacles.append(line)

    @property
    def fires(self) -> List[np.ndarray]:
        """fires is a list of np.ndarray"""
        return self._fires

    @fires.setter
    def fires(self, fires):
        """Input an list of (startx, endx, starty, endy) as start and end of a line"""
        if fires is None:
            self._fires = []
        else:
            self._fires = []
            for startx, endx, starty, endy in fires:
                samples = int(np.linalg.norm((startx - endx, starty - endy)) * self.resolution)
                line = np.array(
                    list(
                        zip(np.linspace(startx, endx, samples), np.linspace(starty, endy, samples))
                    )
                )
                self._fires.append(line)

    @property
    def exits(self) -> List[np.ndarray]:
        """obstacles is a list of np.ndarray"""
        return self._exits

    @exits.setter
    def exits(self, exits):
        """Input an list of (startx, endx, starty, endy) as start and end of a line"""
        if exits is None:
            self._exits = []
        else:
            self._exits = []
            for posx, posy, radius, next_exit in exits:
                self._exits.append(np.array([posx, posy, radius, next_exit]))