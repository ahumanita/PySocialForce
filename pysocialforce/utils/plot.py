"""Utility functions for plots and animations."""

from contextlib import contextmanager

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as mpl_animation
    from matplotlib.patches import Circle, Polygon, Rectangle
    from matplotlib.collections import PatchCollection
except ImportError:
    plt = None
    mpl_animation = None

from .logging import logger
from .stateutils import minmax


@contextmanager
def canvas(image_file=None, **kwargs):
    """Generic matplotlib context."""
    fig, ax = plt.subplots(**kwargs)
    ax.grid(linestyle="dotted")
    ax.set_aspect(1.0, "datalim")
    ax.set_axisbelow(True)

    yield ax

    fig.set_tight_layout(True)
    if image_file:
        fig.savefig(image_file, dpi=300)
    # fig.show()
    plt.close(fig)


@contextmanager
def animation(length: int, movie_file=None, writer=None, **kwargs):
    """Context for animations."""
    fig, ax = plt.subplots(**kwargs)
    fig.set_tight_layout(True)
    ax.grid(linestyle="dotted")
    ax.set_aspect(1.0, "datalim")
    ax.set_axisbelow(True)

    context = {"ax": ax, "update_function": None, "init_function": None}
    yield context

    ani = mpl_animation.FuncAnimation(
        fig,
        init_func=context["init_function"],
        func=context["update_function"],
        frames=length,
        blit=True,
    )
    if movie_file:
        ani.save(movie_file, writer=writer)
    # fig.show()
    plt.close(fig)


class SceneVisualizer:
    """Context for social nav vidualization"""

    def __init__(
        self, scene, output=None, writer="imagemagick", cmap="viridis", agent_colors=None, **kwargs
    ):
        self.scene = scene
        self.states, self.group_states = self.scene.get_states()
        self.cmap = cmap
        self.agent_colors = agent_colors
        self.frames = self.scene.get_length()
        self.output = output
        self.writer = writer

        self.fig, self.ax = plt.subplots(**kwargs)

        self.ani = None

        self.group_actors = None
        self.group_collection = PatchCollection([])
        self.group_collection.set(
            animated=True,
            alpha=0.2,
            cmap=self.cmap,
            facecolors="none",
            edgecolors="purple",
            linewidth=2,
            clip_on=True,
        )

        self.human_actors = None
        self.human_collection = PatchCollection([])
        self.human_collection.set(animated=True, alpha=0.6, cmap=self.cmap, clip_on=True)

    def plot(self):
        """Main method for create plot"""
        self.plot_obstacles()
        self.plot_fires()
        self.plot_exits()
        groups = self.group_states[0]  # static group for now
        if not groups:
            for ped in range(self.scene.peds.size()):
                x = self.states[:, ped, 0]
                y = self.states[:, ped, 1]
                self.ax.plot(x, y, "-o", label=f"ped {ped}", markersize=2.5)
        else:

            colors = plt.cm.rainbow(np.linspace(0, 1, len(groups)))

            for i, group in enumerate(groups):
                for ped in group:
                    x = self.states[:, ped, 0]
                    y = self.states[:, ped, 1]
                    self.ax.plot(x, y, "-o", label=f"ped {ped}", markersize=2.5, color=colors[i])
        self.ax.legend()
        return self.fig

    def animate(self):
        """Main method to create animation"""

        self.ani = mpl_animation.FuncAnimation(
            self.fig,
            init_func=self.animation_init,
            func=self.animation_update,
            frames=self.frames,
            blit=True,
        )

        return self.ani

    def __enter__(self):
        logger.info("Start plotting.")
        self.fig.set_tight_layout(True)
        self.ax.grid(linestyle="dotted")
        self.ax.set_aspect("equal")
        self.ax.margins(2.0)
        self.ax.set_axisbelow(True)
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")

        plt.rcParams["animation.html"] = "jshtml"

        # x, y limit from states, only for animation
        margin = 2.0
        xy_limits = np.array(
            [minmax(state) for state in self.states]
        )  # (x_min, y_min, x_max, y_max)
        xy_min = np.min(xy_limits[:, :2], axis=0) - margin
        xy_max = np.max(xy_limits[:, 2:4], axis=0) + margin
        self.ax.set(xlim=(xy_min[0], xy_max[0]), ylim=(xy_min[1], xy_max[1]))

        # # recompute the ax.dataLim
        # self.ax.relim()
        # # update ax.viewLim using the new dataLim
        # self.ax.autoscale_view()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if exception_type:
            logger.error(
                f"Exception type: {exception_type}; Exception value: {exception_value}; Traceback: {traceback}"
            )
        logger.info("Plotting ends.")
        if self.output:
            if self.ani:
                output = self.output + ".gif"
                logger.info(f"Saving animation as {output}")
                self.ani.save(output, writer=self.writer)
            else:
                output = self.output + ".png"
                logger.info(f"Saving plot as {output}")
                self.fig.savefig(output, dpi=300)
        plt.close(self.fig)

    def plot_human(self, step=-1):
        """Generate patches for human
        :param step: index of state, default is the latest
        :return: list of patches
        """
        states, _ = self.scene.get_states()
        current_state = states[step]
        # radius = 0.2 + np.linalg.norm(current_state[:, 2:4], axis=-1) / 2.0 * 0.3
        radius = [0.2] * current_state.shape[0]
        if self.human_actors:
            for i, human in enumerate(self.human_actors):
                human.center = current_state[i, :2]
                human.set_radius(0.2)
                # human.set_radius(radius[i])
        else:
            self.human_actors = [
                Circle(pos, radius=r) for pos, r in zip(current_state[:, :2], radius)
            ]
        self.human_collection.set_paths(self.human_actors)
        if not self.agent_colors:
            self.human_collection.set_array(np.arange(current_state.shape[0]))
        else:
            # set colors for each agent
            assert len(self.human_actors) == len(
                self.agent_colors
            ), "agent_colors must be the same length as the agents"
            self.human_collection.set_facecolor(self.agent_colors)

    def plot_groups(self, step=-1):
        """Generate patches for groups
        :param step: index of state, default is the latest
        :return: list of patches
        """
        states, group_states = self.scene.get_states()
        current_state = states[step]
        current_groups = group_states[step]
        if self.group_actors:  # update patches, else create
            points = [current_state[g, :2] for g in current_groups]
            for i, p in enumerate(points):
                self.group_actors[i].set_xy(p)
        else:
            self.group_actors = [Polygon(current_state[g, :2]) for g in current_groups]

        self.group_collection.set_paths(self.group_actors)

    def plot_obstacles(self):
        for s in self.scene.get_obstacles():
            self.ax.add_patch(Rectangle((s[:,0][0],s[:,1][0]), s[:,0][-1]-s[:,0][0], s[:,1][-1]-s[:,1][0], fill=True, color="black"))

    def plot_fires(self):
        for f in self.scene.get_fires():
            self.ax.add_patch(Rectangle((f[:,0][0],f[:,1][0]), f[:,0][-1]-f[:,0][0], f[:,1][-1]-f[:,1][0], fill=True, color="red"))

    def plot_exits(self):
        for e in self.scene.get_exits():
            self.ax.add_patch(Circle((e[0],e[1]), e[2], fill=True, color="green", alpha=0.2))

    def animation_init(self):
        self.plot_obstacles()
        self.plot_fires()
        self.plot_exits()
        self.ax.add_collection(self.group_collection)
        self.ax.add_collection(self.human_collection)

        return (self.group_collection, self.human_collection)

    def animation_update(self, i):
        self.plot_groups(i)
        self.plot_human(i)
        return (self.group_collection, self.human_collection)

    def plot_escaped(self) :
        fig, ax = plt.subplots()
        escaped = [i/self.scene.peds.get_nr_peds()*100 for i in self.scene.peds.escaped]
        timesteps = [t for t in range(len(escaped))]
        ax.plot(timesteps,escaped)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Escaped people [%]")
        ax.set_title("Percent of people that are escaped the building in total over time.")
        fig.savefig(self.output + "_escaped.png")
        logger.info("Created plot of escaped people.")