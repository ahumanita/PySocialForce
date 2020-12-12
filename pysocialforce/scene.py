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
        self.follower_radius = config("follower_radius", 2.0)
        self.enable_following = config("enable_following", False)
        self.smoke_change = config("smoke_change", 0.01)
        self.panic_change_s = config("panic_change_s", 0.01)
        self.panic_change_t = config("panic_change_t", 0.001)
        self.health_change = config("health_change", 0.01)
        self.esc_fir = config("esc_fire", 0.01)
        self.simulator = simulator

        self.border = border

        self.max_speeds = None
        self.initial_speeds = None

        self.ped_states = []
        self.group_states = []
        self.escaped = []
        self.smoke_radii = [self.simulator.get_smoke_radius()]
        self.av_health = []
        self.av_panic = []
        self.dead = []

        self.update(state, groups)

    def update(self, state, groups):
        self.state = state
        self.groups = groups
        self.escaped.append(self.get_nr_escaped())
        self.av_health.append(np.sum(self.state[:,10])/self.state.shape[0])
        self.av_panic.append(np.sum(self.state[:,11])/self.state.shape[0])
        self.dead.append(np.sum(self.state[:,10] == 0))

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        tau = self.default_tau * np.ones(state.shape[0])
        if state.shape[1] < 7:
            # escaped: index 7
            escaped = np.zeros(state.shape[0])
            # target exit: index 8
            targeted_exit = -np.ones(state.shape[0])
            # smoke coefficient: index 9
            smoke = np.zeros(state.shape[0])
            # health coefficient: index 10
            health = np.ones(state.shape[0])
            # panic coefficient: index 11
            panic = np.zeros(state.shape[0])
            # extend states
            state = np.concatenate((state, np.expand_dims(tau, -1)), axis=-1)
            state = np.concatenate((state, np.expand_dims(escaped, -1)), axis=-1)
            state = np.concatenate((state, np.expand_dims(targeted_exit, -1)), axis=-1)
            state = np.concatenate((state, np.expand_dims(smoke, -1)), axis=-1)
            state = np.concatenate((state, np.expand_dims(health, -1)), axis=-1)
            self._state = np.concatenate((state, np.expand_dims(panic, -1)), axis=-1)
        else:
            self._state = state
        if self.initial_speeds is None:
            self.initial_speeds = self.speeds()
        self.max_speeds = self.max_speed_multiplier * self.initial_speeds
        self.ped_states.append(self._state.copy())

    def get_states(self):
        return np.stack(self.ped_states), self.group_states

    def get_ped_states(self):
        return self.ped_states

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
        exit_id = int(exit_id)
        self.set_goal(p, self.simulator.get_exits()[exit_id,:2])
        self.state[p,8] = exit_id

    def tau(self):
        return self.state[:, 6:7]

    def escaped(self) :
        return self.state[:,7]

    def targeted_exit(self) :
        return self.state[:,8]

    def get_nr_escaped(self) :
        return np.sum(self.state[:,7])

    def get_nr_dead(self) :
        return self.dead[-1]

    def get_smoke_radii(self) :
        return self.smoke_radii
    
    def get_nr_peds(self) :
        return self.state.shape[0]

    def speeds(self):
        """Return the speeds corresponding to a given state."""
        return stateutils.speeds(self.state)

    def update_target(self, next_state) :
        # Update the directions of those that don't follow an exit yet
        if self.simulator.get_exits() is not None :
            not_to_exit = np.where(next_state[:,8] == -1)[0]
            if not_to_exit.size > 0 :
                if self.simulator.get_exits() is not None :
                    exits = self.simulator.get_exits()
                    for p in np.nditer(not_to_exit) :
                        p_pos = next_state[p,:2]
                        # Set exit as target when person is in exit radius
                        to_exit = False
                        for e in range(exits.shape[0]) :
                            if np.linalg.norm(p_pos-exits[e,:2]) < exits[e,2] :
                                self.set_exit_goal(p,e)
                                to_exit = True
                        if to_exit :
                            continue
                        # Otherwise set target to opposite of fire
                        if self.simulator.get_fires() is not None :
                            f = self.simulator.get_fires()[0]
                            fire_center = np.array([f[:,0][0]+(f[:,0][-1]-f[:,0][0])/2, f[:,1][0]+(f[:,1][-1]-f[:,1][0])/2])
                            target = p_pos - (fire_center - p_pos)
                            self.simulator.peds.set_goal(p, target)

    def follower_model(self, next_state) :
        # Qiu, 2009: Following people get as direction the average direction of the other group members
        # As we do not have any groups yet so maybe just define a radius and average the direction of 
        # the other people inside the radius?
        if self.enable_following :
            not_to_exit = np.where(next_state[:,8] == -1)[0]
            if not_to_exit.size > 0 :
                np.random.shuffle(not_to_exit)
                for p in np.nditer(not_to_exit) :
                    p_pos = next_state[p,:2]
                    average_direction = np.array([0.0,0.0])
                    nneighbors = 0
                    for n in range(next_state.shape[0]) :
                        if np.linalg.norm(next_state[n,:2] - p_pos) <= self.follower_radius :
                            average_direction += next_state[n,2:4]
                            nneighbors += 1
                    target = next_state[p,4:6]
                    if nneighbors > 0 :
                        average_direction /= nneighbors
                        target = stateutils.turn_vector_around_other(target, p_pos, average_direction)
                    self.set_goal(p,target)  

    def step(self, force, groups=None):
        """Move peds according to forces"""   
        next_state = self.state
        desired_velocity = self.vel() #+ self.step_width * force
        # Avoid fire
        if self.simulator.get_fires() is not None :
            # Get position opposite to fire
            f = self.simulator.get_fires()[0]
            fire_center = np.array([f[:,0][0]+(f[:,0][-1]-f[:,0][0])/2, f[:,1][0]+(f[:,1][-1]-f[:,1][0])/2])
            opp_fire_dir = - (fire_center - next_state[:,:2])
            target_dir = desired_velocity
            # Get angles
            target_angle = np.mod(np.arctan2(target_dir[:,1],target_dir[:,0]),2*np.pi)
            fire_angle = np.mod(np.arctan2(opp_fire_dir[:,1],opp_fire_dir[:,0]),2*np.pi)
            # Average angles and give weight of smoke impact
            theta = np.zeros(len(target_angle))
            index1 = np.where(np.abs(target_angle-fire_angle)>= np.pi)
            index2 = np.where(np.abs(target_angle-fire_angle)< np.pi)
            esc_fir = self.esc_fir
            theta[index1] = ((1-esc_fir-(1-esc_fir)*next_state[index1, 9])*target_angle[index1]+(esc_fir+(1-esc_fir)*next_state[index1, 9])*fire_angle[index1]) - 2*np.pi*(1-esc_fir)
            theta[index2] = ((1-esc_fir-(1-esc_fir)*next_state[index2, 9])*target_angle[index2]+(esc_fir+(1-esc_fir)*next_state[index2, 9])*fire_angle[index2])
        # Add random angle due to smoke and panic
        theta += (next_state[:, 9] + next_state[:,11])*np.random.uniform(-np.pi,np.pi,len(desired_velocity))
        L_vel=np.sqrt(desired_velocity[:,0]**2+desired_velocity[:,1]**2)
        # Update next position
        desired_velocity[:,0] = L_vel*np.cos(theta)
        desired_velocity[:,1] = L_vel*np.sin(theta)
        # desired velocity
        desired_velocity = desired_velocity + self.step_width * force
        desired_velocity = self.capped_velocity(desired_velocity, self.max_speeds)
        # stop when arrived
        desired_velocity[stateutils.desired_directions(self.state)[1] < 0.5] = [0, 0]        
        # update state
        next_state[:, 0] += desired_velocity[:,0] * self.step_width * next_state[:, 10] * (1+next_state[:, 11])
        next_state[:, 1] += desired_velocity[:,1] * self.step_width * next_state[:, 10] * (1+next_state[:, 11])
        next_state[:, 2:4] = desired_velocity
        next_groups = self.groups
        if self.border is not None :
            escaped_rows = np.where((next_state[:,0] <= self.border[0]) | (next_state[:,0] >= self.border[1]) | (next_state[:,1] <= self.border[2]) | (next_state[:,1] >= self.border[3]))[0]
            if escaped_rows.size > 0 :
                next_state[escaped_rows,7] = 1
        # TODO: efficient way to do this? 
        # Update target to next exit when agent is close enough
        if self.simulator.get_exits() is not None :
            exits = self.simulator.get_exits()
            exit_as_target = np.where(next_state[:,8] >= 0)[0]
            if exit_as_target.size > 0 :
                for p in np.nditer(exit_as_target) :
                    if np.linalg.norm(next_state[p,:2]-exits[int(next_state[p,8]),:2]) < 2*self.agent_radius :
                        next_exit_id = exits[int(next_state[p,8]),3]
                        if next_exit_id is not None :
                            self.set_exit_goal(p,next_exit_id)
        # Update the directions of those that don't follow an exit yet
        self.update_target(next_state)
        # Qiu, 2009: Following people get as direction the average direction of the other group members
        self.follower_model(next_state)               

        # Smoke and panic over time
        if self.simulator.get_fires() is not None : 
            f = self.simulator.get_fires()[0]
            next_state[:,11] += self.panic_change_t
            next_state, new_smoke_radius = stateutils.smoke(next_state,
                                                self.simulator.get_smoke_radius(),
                                                f,
                                                self.smoke_change,
                                                self.health_change,
                                                self.panic_change_s)
            self.simulator.set_smoke_radius(new_smoke_radius)
            self.smoke_radii.append(new_smoke_radius)

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
                closest_dist = np.linalg.norm(pos[p]-exits[0,:2])
                for e in range(1,len(exits)) :
                    dist = np.linalg.norm(pos[p]-exits[e,:2])
                    if dist < closest_dist :
                        closest = e
                        closest_dist = dist
                self.set_exit_goal(p,closest)

    # Walk in opposite direction to fire
    def set_fire_target(self) :
        f = self.simulator.get_fires()[0]
        fire_center = np.array([f[:,0][0]+(f[:,0][-1]-f[:,0][0])/2, f[:,1][0]+(f[:,1][-1]-f[:,1][0])/2])
        states = self.simulator.peds.get_ped_states()[0]
        for p in range(self.get_nr_peds()) :
            p_pos = states[p,0:2]
            target = p_pos - (fire_center - p_pos)
            self.simulator.peds.set_goal(p, target)

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
        self.smoke_radius = 0.01

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
        self._fires = None
        if fires is not None:
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
        self._exits = None
        if exits is not None:
            self._exits = []
            for posx, posy, radius, next_exit in exits:
                self._exits.append(np.array([posx, posy, radius, next_exit]))
            self._exits = np.array(self._exits)