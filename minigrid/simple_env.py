from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Goal
from minigrid.core.grid import Grid
from gymnasium.spaces import Text


class SimpleEnv(MiniGridEnv):
    def __init__(self, grid_size=6, render_mode="human"):
        mission_space = Text(max_length=1)  # dummy mission space
        super().__init__(
            grid_size=grid_size,
            max_steps=100,
            render_mode=render_mode,
            mission_space=mission_space
        )
        self.agent_start_pos = (1, 1)
        self.agent_start_dir = 0

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Add walls around the border
        self.grid.wall_rect(0, 0, width, height)

        # Place the goal at bottom-right (inside walls)
        self.put_obj(Goal(), width - 2, height - 2)

        # Set agent starting position and direction
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir
        self.mission = ""  # set a dummy mission string
