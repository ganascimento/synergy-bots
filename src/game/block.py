import pygame
import utils.config as config


class Block(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface([config.CELL_SIZE, config.CELL_SIZE])
        self.image.fill(config.OBSTACLE_COLOR)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def normalized_x(self):
        return self.rect.x // config.CELL_SIZE

    def normalized_y(self):
        return self.rect.y // config.CELL_SIZE
