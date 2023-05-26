'''entities.py'''
class Entity:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Pacman(Entity):
    def __init__(self, x, y, lives):
        super().__init__(x, y)
        self.score = 0
        self.lives = lives

    def move(self, dx, dy):
        self.x += dx
        self.y += dy


class Ghost:
    def __init__(self, x, y, difficulty=0):
        self.x = x
        self.y = y
        self.start_x = x
        self.start_y = y
        self.difficulty = difficulty


class Pellet(Entity):
    def __init__(self, x, y):
        super().__init__(x, y)


class PowerPellet(Pellet):
    def __init__(self, x, y):
        super().__init__(x, y)
