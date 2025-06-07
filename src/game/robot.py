import pygame
import utils.config as config
import numpy as np
from typing import List, Tuple


class Robot(pygame.sprite.Sprite):
    def __init__(self, x: int, y: int, color: Tuple[int, int, int], blocks_group: pygame.sprite.Group):
        super().__init__()
        self.image = pygame.Surface([config.CELL_SIZE, config.CELL_SIZE])
        self.color = color
        self.image.fill(self.color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

        self.blocks_group = blocks_group
        self.clear_cells: List[Tuple[int, int]] = []

    def reset_internal_state(self):
        self.clear_cells = []

    def step(
        self,
        action: int,
        all_robots_list: List["Robot"],
        total_clearable_cells: int,
    ) -> Tuple[np.ndarray, float]:
        all_marks = []
        for r in all_robots_list:
            all_marks.extend(r.clear_cells)

        all_marks = list(set(all_marks))
        other_robots_group = pygame.sprite.Group([r for r in all_robots_list if r != self])

        reward = -0.1

        action_map = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
            4: (0, 0),
        }

        dx_norm, dy_norm = action_map.get(action, (0, 0))
        dx = dx_norm * config.CELL_SIZE
        dy = dy_norm * config.CELL_SIZE

        original_x, original_y = self.rect.x, self.rect.y
        self.rect.x += dx
        self.rect.y += dy

        collided_obstacle = pygame.sprite.spritecollideany(self, self.blocks_group)
        collided_robot = pygame.sprite.spritecollideany(self, other_robots_group)
        out_of_bounds = (
            self.rect.left < 0 or self.rect.right > config.WIDTH or self.rect.top < 0 or self.rect.bottom > config.HEIGHT
        )

        valid_move = True
        if collided_obstacle:
            reward -= 0.1
            self.rect.x, self.rect.y = original_x, original_y
            valid_move = False
        elif collided_robot:
            self.rect.x, self.rect.y = original_x, original_y
            reward -= 0.2
            valid_move = False
        elif out_of_bounds:
            reward -= 0.1
            self.rect.x, self.rect.y = original_x, original_y
            valid_move = False

        if valid_move:
            position = (self.rect.x, self.rect.y)

            if position not in all_marks:
                self.clear_cells.append(position)
                all_marks.append(position)
                num_total_cleaned_cells = len(all_marks)

                reward += 1.5

                if num_total_cleaned_cells == total_clearable_cells:
                    reward += 8

        next_local_state = self.get_state(all_robots_list, all_marks)

        return next_local_state, reward

    def normalized_x(self) -> int:
        return self.rect.x // config.CELL_SIZE

    def normalized_y(self) -> int:
        return self.rect.y // config.CELL_SIZE

    def get_state(self, all_robots_list: pygame.sprite.Group, all_cleaned_pixels=None) -> np.ndarray:
        map_grid_width = config.GRID_WIDTH
        map_grid_height = config.GRID_HEIGHT
        if all_cleaned_pixels is None:
            all_cleaned_pixels = []
            for r in all_robots_list:
                all_cleaned_pixels.extend(r.clear_cells)

        all_cleaned_pixels = list(set(all_cleaned_pixels))
        other_robots_group = [r for r in all_robots_list if r != self]
        dirt_grid = np.ones((map_grid_height, map_grid_width), dtype=np.float32)

        for pixel_x, pixel_y in all_cleaned_pixels:
            grid_x = pixel_x // config.CELL_SIZE
            grid_y = pixel_y // config.CELL_SIZE
            if 0 <= grid_y < map_grid_height and 0 <= grid_x < map_grid_width:
                dirt_grid[grid_y, grid_x] = 0.0

        for block in self.blocks_group:
            norm_y = block.normalized_y()
            norm_x = block.normalized_x()
            if 0 <= norm_y < map_grid_height and 0 <= norm_x < map_grid_width:
                dirt_grid[norm_y, norm_x] = -1.0

        flattened_dirt_map = dirt_grid.flatten()

        norm_self_pos_y = self.normalized_y() / map_grid_height
        norm_self_pos_x = self.normalized_x() / map_grid_width
        normalized_self_position = np.array([norm_self_pos_y, norm_self_pos_x], dtype=np.float32)

        ally_positions_list = []
        num_expected_allies = config.ROBOT_NUMBER - 1

        for i in range(num_expected_allies):
            norm_ally_pos_y = other_robots_group[i].normalized_y() / map_grid_height
            norm_ally_pos_x = other_robots_group[i].normalized_x() / map_grid_width
            ally_positions_list.extend([norm_ally_pos_y, norm_ally_pos_x])

        normalized_ally_positions = np.array(ally_positions_list, dtype=np.float32)

        state_vector = np.concatenate((flattened_dirt_map, normalized_self_position, normalized_ally_positions))

        return state_vector
