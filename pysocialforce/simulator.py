# coding=utf-8

"""Synthetic pedestrian behavior with social groups simulation according to the Extended Social Force model.

See Helbing and Molnár 1998 and Moussaïd et al. 2010
"""
from pysocialforce.utils import DefaultConfig
from pysocialforce.scene import PedState, EnvState
from pysocialforce import forces
import numpy as np


class Simulator:
    """Simulate social force model.

    ...

    Attributes
    ----------
    state : np.ndarray [n, 6] or [n, 7]
       Each entry represents a pedestrian state, (x, y, v_x, v_y, d_x, d_y, [tau])
    obstacles : np.ndarray
        Environmental obstacles
    groups : List of Lists
        Group members are denoted by their indices in the state
    config : Dict
        Loaded from a toml config file
    max_speeds : np.ndarray
        Maximum speed of pedestrians
    forces : List
        Forces to factor in during navigation

    Methods
    ---------
    capped_velocity(desired_velcity)
        Scale down a desired velocity to its capped speed
    step()
        Make one step
    """

    def __init__(self, state, groups=None, obstacles=None, fires=None, exits=None, border=None, config_file=None):
        self.config = DefaultConfig()
        if config_file:
            self.config.load_config(config_file)
        # TODO: load obstacles from config
        self.scene_config = self.config.sub_config("scene")
        # initiate obstacles
        self.env = EnvState(self.scene_config, obstacles, fires, exits, self.config("resolution", 10.0))

        # initiate agents
        self.peds = PedState(self, state, groups, border, self.scene_config)
        # if the agents have knowledge about the positions of emergency exits,
        # set closest emergency exit as target
        if self.scene_config("exit_knowledge") :
            self.peds.set_exits_as_targets()
        else :
            # We only want to consider one fire right now..
            if fires is not None and len(fires) <= 1 :
                self.peds.set_fire_target()

        # construct forces
        self.forces = self.make_forces(self.config)

    def make_forces(self, force_configs):
        """Construct forces"""
        force_list = [
            forces.DesiredForce(),
            forces.SocialForce(),
            forces.ObstacleForce(),
            forces.FireForce(),
            forces.PedRepulsiveForce(),
            forces.SpaceRepulsiveForce(),
        ]
        group_forces = [
            forces.GroupCoherenceForceAlt(),
            forces.GroupRepulsiveForce(),
            forces.GroupGazeForceAlt(),
        ]
        if self.scene_config("enable_group"):
            force_list += group_forces

        # initiate forces
        for force in force_list:
            force.init(self, force_configs)

        return force_list

    def compute_forces(self):
        """compute forces"""
        return sum(map(lambda x: x.get_force(), self.forces))

    def get_states(self):
        """Expose whole state"""
        return self.peds.get_states()

    def get_length(self):
        """Get simulation length"""
        return len(self.get_states()[0])

    def get_obstacles(self):
        return self.env.obstacles

    def get_fires(self) :
        return self.env.fires

    def get_exits(self) :
        return self.env.exits

    def get_smoke_radius(self) :
        return self.env.smoke_radius

    def set_smoke_radius(self, new_radius) :
        self.env.smoke_radius = new_radius

    def get_escaped(self) :
        return np.array(self.peds.escaped)

    def get_dead(self) :
        return np.array(self.peds.dead)

    def get_health(self) :
        return np.array(self.peds.av_health)

    def get_panic(self) :
        return np.array(self.peds.av_panic)

    def step_once(self):
        """step once"""
        self.peds.step(self.compute_forces())

    def step(self, n=1):
        """Step n time"""
        N = n
        finished = False
        for _ in range(n):
            self.step_once()
            if not finished :
                if self.peds.get_nr_escaped() + self.peds.get_nr_dead() == self.peds.get_nr_peds() :
                    print(str(self.peds.get_nr_escaped()) + " escaped and " + str(self.peds.get_nr_dead()) + " died of " + str(self.peds.get_nr_peds()) + " in total after " + str(_) + " steps.")
                    finished = True
                    N = _
        return N

    def step_until_end(self):
        all_escaped_or_dead = False
        n = 1
        N = 1
        while not all_escaped_or_dead :
            self.step_once()
            if self.peds.get_nr_escaped() == self.peds.get_nr_peds() :
                print("All people escaped after " + str(n) + " steps!")
                all_escaped_or_dead = True
                N = n
                return N
            if self.peds.get_nr_dead() == self.peds.get_nr_peds() :
                print("All people died after " + str(n) + " steps!")
                all_escaped_or_dead = True
                N = n
                return N
            if self.peds.get_nr_escaped() + self.peds.get_nr_dead() == self.peds.get_nr_peds() :
                print(str(self.peds.get_nr_escaped()) + " escaped and " + str(self.peds.get_nr_dead()) + " died of " + str(self.peds.get_nr_peds()) + " in total after " + str(n) + " steps.")
                all_escaped_or_dead = True
                N = n
                return N
            n += 1
        return n