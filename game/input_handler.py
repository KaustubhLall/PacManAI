''' input_handler.py'''
import pygame


class InputHandler:
    def __init__(self, manual_mode=False):
        self.manual_mode = manual_mode

    def get_input(self):
        dx, dy = 0, 0
        if self.manual_mode:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                dx, dy = 0, -1
            elif keys[pygame.K_DOWN]:
                dx, dy = 0, 1
            elif keys[pygame.K_LEFT]:
                dx, dy = -1, 0
            elif keys[pygame.K_RIGHT]:
                dx, dy = 1, 0
        return dx, dy
