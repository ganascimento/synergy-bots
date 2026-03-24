import pygame
import random
from collections import deque
import utils.config as config
import numpy as np
from .block import Block
from .robot import Robot
from .clear_mark import ClearMark
from typing import List, Tuple


class Game:
    def __init__(self):
        self.screen: pygame.Surface = None
        self.clock: pygame.time.Clock = None
        self.ready: bool = False

        self.blocks: pygame.sprite.Group = pygame.sprite.Group()
        self.robots: pygame.sprite.Group = pygame.sprite.Group()
        self.all_robots_list: List[Robot] = []

        self.step_count: int = 0
        self._total_clearable_cells_on_reset: int = 0

        self.grid_width = config.WIDTH // config.CELL_SIZE
        self.grid_height = config.HEIGHT // config.CELL_SIZE

    def reset(self) -> None:
        self.blocks = self._generate_room(config.CELL_SIZE)
        self.all_robots_list = self._initialize_robots(config.ROBOT_NUMBER, config.CELL_SIZE)
        self.robots.empty()
        self.robots.add(self.all_robots_list)

        self.step_count = 0
        self._total_clearable_cells_on_reset = self._calculate_total_clearable_cells()

    def step(self, joint_actions: List[int]) -> Tuple[List[np.ndarray], List[float], bool]:
        self.step_count += 1

        next_local_states: List[np.ndarray] = []
        rewards: List[float] = []

        for i, robot_instance in enumerate(self.all_robots_list):
            action = joint_actions[i]

            next_state, reward = robot_instance.step(action, self.all_robots_list, self._total_clearable_cells_on_reset)
            next_local_states.append(next_state)
            rewards.append(reward)

        done = self.is_done() or self.step_count >= config.MAX_STEPS
        return next_local_states, rewards, [done] * config.ROBOT_NUMBER, self.step_count

    def play_render(self) -> None:
        if not self.ready:
            self._prepare_pygame()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.clock.tick(config.FPS)
        self.screen.fill((255, 255, 255))

        self.blocks.draw(self.screen)
        self.robots.draw(self.screen)
        self._draw_clear_marks_render()

        pygame.display.flip()

    def is_done(self) -> bool:
        current_grid_state = self.get_grid_state()
        return np.sum(current_grid_state == 0) == 0

    def get_grid_state(self) -> np.ndarray:
        grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)

        for block in self.blocks:
            norm_y = block.rect.y // config.CELL_SIZE
            norm_x = block.rect.x // config.CELL_SIZE
            if 0 <= norm_y < self.grid_height and 0 <= norm_x < self.grid_width:
                grid[norm_y, norm_x] = 1

        for robot_instance in self.all_robots_list:
            for pos_x_px, pos_y_px in robot_instance.clear_cells:
                norm_y = pos_y_px // config.CELL_SIZE
                norm_x = pos_x_px // config.CELL_SIZE
                if 0 <= norm_y < self.grid_height and 0 <= norm_x < self.grid_width:
                    grid[norm_y, norm_x] = 1
        return grid

    def _calculate_total_clearable_cells(self) -> int:
        obstacle_map = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
        for block in self.blocks:
            norm_y = block.rect.y // config.CELL_SIZE
            norm_x = block.rect.x // config.CELL_SIZE
            if 0 <= norm_y < self.grid_height and 0 <= norm_x < self.grid_width:
                obstacle_map[norm_y, norm_x] = 1

        return np.sum(obstacle_map == 0)

    def get_total_clearable_cells_on_reset(self) -> int:
        return self._total_clearable_cells_on_reset

    def count_uncleaned_cells(self) -> int:
        current_grid_state = self.get_grid_state()
        num_uncleaned = np.sum(current_grid_state == 0)
        return int(num_uncleaned)

    def _prepare_pygame(self) -> None:
        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Robôs de Limpeza Cooperativos")
        self.ready = True

    def _generate_room(self, cell_size: int) -> pygame.sprite.Group:
        while True:
            obstacle_map = np.zeros((self.grid_height, self.grid_width), dtype=bool)

            for row in range(self.grid_height):
                for col in range(self.grid_width):
                    if random.random() < config.OBSTACLE_PROBABILITY:
                        obstacle_map[row, col] = True

            if self._is_fully_connected(obstacle_map):
                blocks_group = pygame.sprite.Group()
                for row in range(self.grid_height):
                    for col in range(self.grid_width):
                        if obstacle_map[row, col]:
                            blocks_group.add(Block(col * cell_size, row * cell_size))
                return blocks_group

    def _is_fully_connected(self, obstacle_map: np.ndarray) -> bool:
        """BFS flood-fill: returns True only if every free cell is reachable from every other free cell."""
        free_cells = [
            (r, c)
            for r in range(self.grid_height)
            for c in range(self.grid_width)
            if not obstacle_map[r, c]
        ]
        if not free_cells:
            return False

        visited = set()
        queue = deque([free_cells[0]])
        visited.add(free_cells[0])

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited and 0 <= nr < self.grid_height and 0 <= nc < self.grid_width and not obstacle_map[nr, nc]:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        return len(visited) == len(free_cells)

    def _initialize_robots(self, num_robots: int, cell_size: int) -> List[Robot]:
        temp_robots_list: List[Robot] = []
        current_robot_sprites = pygame.sprite.Group()

        for i in range(num_robots):
            while True:
                x_norm = random.randrange(0, self.grid_width)
                y_norm = random.randrange(0, self.grid_height)
                x_px = x_norm * cell_size
                y_px = y_norm * cell_size

                robot_color = config.ROBOT_COLORS[i % len(config.ROBOT_COLORS)]
                temp_robot = Robot(x_px, y_px, robot_color, self.blocks)

                collides_block = pygame.sprite.spritecollideany(temp_robot, self.blocks)
                collides_other_robot = pygame.sprite.spritecollideany(temp_robot, current_robot_sprites)

                if not collides_block and not collides_other_robot:
                    temp_robots_list.append(temp_robot)
                    current_robot_sprites.add(temp_robot)
                    break

        return temp_robots_list

    def _draw_clear_marks_render(self) -> None:
        clear_marks_group = pygame.sprite.Group()
        for robot_instance in self.all_robots_list:
            for x_px, y_px in robot_instance.clear_cells:
                clear_mark = ClearMark(x_px, y_px, robot_instance.color)
                clear_marks_group.add(clear_mark)
        clear_marks_group.draw(self.screen)

    def close(self):
        if self.ready:
            pygame.quit()
            self.ready = False
