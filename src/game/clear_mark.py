import pygame
import utils.config as config


class ClearMark(pygame.sprite.Sprite):
    def __init__(self, x, y, color):
        super().__init__()
        self.image = pygame.Surface([config.MARK_SIZE, config.MARK_SIZE])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = int(x + (config.CELL_SIZE - config.MARK_SIZE) / 2)
        self.rect.y = int(y + (config.CELL_SIZE - config.MARK_SIZE) / 2)
