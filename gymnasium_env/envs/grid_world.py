from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3

class TileType(Enum):
    EMPTY = 0
    WALL = 1
    PIT = 2
    A = 3
    B = 4

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.right.value: np.array([0, 1]),
            Actions.up.value: np.array([-1, 0]),
            Actions.left.value: np.array([0, -1]),
            Actions.down.value: np.array([1, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self.grid = np.full((self.size, self.size), TileType.EMPTY, dtype=object)
        self._place_tiles()
        self.visited_A = False
        self.visited_B = False

    def _place_tiles(self):
        self.walls = [
            np.array([0,1]), np.array([1,3]), np.array([3,1]), np.array([4,2]), np.array([3,4]) 
        ]
        for r,c in self.walls:
            self.grid[r,c] = TileType.WALL

        self.pits = [
            np.array([1,1]), np.array([3,2]), np.array([1,4])
        ]
        for r,c in self.pits:
            self.grid[r,c] = TileType.PIT
        
        self.tile_A = (0,4)
        self.tile_B = (4,0)


        self.grid[self.tile_A] = TileType.A
        self.grid[self.tile_B] = TileType.B

        self._init_agent_location = np.array([0, 0])
        self._target_location = np.array([4, 4])


    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        #self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._agent_location = np.array(self._init_agent_location)
        self.visited_A = False
        self.visited_B = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        terminated = False
        reward = -0.1

        new_agent_location = self._agent_location + self._action_to_direction[action]
        if not (0 <= new_agent_location[0] < self.size and 0 <= new_agent_location[1] < self.size):
            return self._get_obs(), reward,  terminated, False, self._get_info()
        if self.grid[new_agent_location[0], new_agent_location[1]] == TileType.WALL:
            return self._get_obs(), reward,  terminated, False, self._get_info()
        else:
            self._agent_location = new_agent_location
        current_tile = self.grid[self._agent_location[0], self._agent_location[1]]

        if np.array_equal(self._agent_location, self._target_location):
            reward += 100
            terminated = True

        if current_tile == TileType.PIT:
            reward += -10
            terminated = True
        elif current_tile == TileType.A:
            if not self.visited_A:
                reward += 10
                self.visited_A = True
        elif current_tile == TileType.B and not self.visited_B:
            reward += 10
            self.visited_B = True
        elif current_tile == TileType.B and self.visited_B:    
            terminated = True
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Fill canva with specialtiles tiles:
        for r in range(self.size):
            for c in range(self.size):
                tile_type = self.grid[r,c]
                color=(255,255,255)
                if tile_type == TileType.WALL:
                    color = (175, 76, 15)
                elif tile_type == TileType.PIT:
                    color = (115, 115, 115)
                elif tile_type == TileType.A:
                    color = (126, 217, 87)
                elif tile_type == TileType.B:
                    color = (255, 222, 89)
                
                pygame.draw.rect(
                    canvas,
                    color, 
                    pygame.Rect(
                        pix_square_size*np.array([c,r]),
                        (pix_square_size, pix_square_size)
                        )
                )
        # First we draw the target
        target_x = int((self._target_location[1] + 0.5) * pix_square_size)
        target_y = int((self._target_location[0] + 0.5) * pix_square_size)

        pygame.draw.circle(
            canvas,
            (255, 0,0),
            (target_x, target_y),
            int(pix_square_size / 2),
        )
        
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (int((self._agent_location[1] + 0.5) * pix_square_size), int((self._agent_location[0] + 0.5) * pix_square_size)),
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas), dtype=np.uint8), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
