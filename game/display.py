'''display.py'''
import pygame

# Top-level configuration
CONFIG = {
    'button': {
        'width': 150,
        'height': 40,
        'colors': {
            'background': (200, 200, 200),
            'text': (0, 0, 0)
        }
    },
    'colors': {
        'background': (0, 0, 0),
        'wall': (100, 100, 100),
        'pellet': (255, 255, 255),
        'pacman': (255, 255, 0),
        'ghost': (255, 0, 0),
        'score': (255, 255, 255),
        'lives': (255, 255, 255)
    },
    'font_size': 36
}

class Button:
    def __init__(self, x, y, text, callback):
        self.x = x
        self.y = y
        self.text = text
        self.callback = callback

    def draw(self, screen):
        pygame.draw.rect(screen, CONFIG['button']['colors']['background'],
                         (self.x, self.y, CONFIG['button']['width'], CONFIG['button']['height']))
        font = pygame.font.Font(None, CONFIG['font_size'])
        text = font.render(self.text, True, CONFIG['button']['colors']['text'])
        screen.blit(text, (self.x + 10, self.y + 10))

    def handle_click(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            if self.x < mouse_pos[0] < self.x + CONFIG['button']['width'] and \
               self.y < mouse_pos[1] < self.y + CONFIG['button']['height']:
                self.callback()

class Display:
    def __init__(self, game_state):
        self.game_state = game_state
        pygame.init()

        infoObject = pygame.display.Info()
        self.screen_width = infoObject.current_w
        self.screen_height = infoObject.current_h

        self.cell_size = min(self.screen_width // game_state.board_width,
                             self.screen_height // (game_state.board_height + 2))

        self.screen = pygame.display.set_mode(
            (game_state.board_width * self.cell_size,
             (game_state.board_height + 2) * self.cell_size))

        toolbar_width = self.screen_width
        toolbar_height = 60
        self.toolbar_rect = pygame.Rect(0, 0, toolbar_width, toolbar_height)

        self.buttons = [
            Button(10, 10, "Easy", lambda: self.change_difficulty(0)),
            Button(170, 10, "Medium", lambda: self.change_difficulty(1)),
            Button(330, 10, "Hard", lambda: self.change_difficulty(2)),
            Button(490, 10, "Insane", lambda: self.change_difficulty(3)),
            Button(self.screen_width - 160, 10, "Exit", pygame.quit)
        ]

    def change_difficulty(self, difficulty):
        for ghost in self.game_state.ghosts:
            ghost.difficulty = difficulty
        print(f"Changing difficulty to {difficulty}")

    def draw(self):
        self.screen.fill(CONFIG['colors']['background'])

        for y in range(self.game_state.board_height):
            for x in range(self.game_state.board_width):
                rect = pygame.Rect(x * self.cell_size, (y + 2) * self.cell_size,
                                   self.cell_size, self.cell_size)
                if self.game_state.board[y][x] == '#':
                    pygame.draw.rect(self.screen, CONFIG['colors']['wall'], rect)

        for pellet in self.game_state.pellets:
            pygame.draw.circle(self.screen, CONFIG['colors']['pellet'],
                               (pellet.x * self.cell_size + self.cell_size // 2,
                                (pellet.y + 2) * self.cell_size + self.cell_size // 2),
                               self.cell_size // 4)

        if self.game_state.pacman:
            rect = pygame.Rect(self.game_state.pacman.x * self.cell_size,
                               (self.game_state.pacman.y + 2) * self.cell_size,
                               self.cell_size, self.cell_size)
            pygame.draw.ellipse(self.screen, CONFIG['colors']['pacman'], rect)

        for ghost in self.game_state.ghosts:
            rect = pygame.Rect(ghost.x * self.cell_size, (ghost.y + 2) * self.cell_size,
                               self.cell_size, self.cell_size)
            pygame.draw.ellipse(self.screen, CONFIG['colors']['ghost'], rect)

        self.draw_text('Score', self.game_state.pacman.score, (800, 10))
        self.draw_text('Lives', self.game_state.pacman.lives, (800, 50))

        for button in self.buttons:
            button.draw(self.screen)

        pygame.display.flip()

    def draw_text(self, label, value, pos):
        font = pygame.font.Font(None, CONFIG['font_size'])
        text = font.render(f'{label}: {value}', True, CONFIG['colors'][label.lower()])
        text_rect = text.get_rect()
        text_rect.topright = pos
        self.screen.blit(text, text_rect)


    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            for button in self.buttons:
                button.handle_click(event)
