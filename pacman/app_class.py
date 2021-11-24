import pygame, sys
import copy

from pygame.event import event_name
from settings import *
from player_class import *
from enemy_class import *

pygame.init()

vec = pygame.math.Vector2


class App:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True
        self.state = 'start'
        self.cell_width = MAZE_WIDTH//COLS
        self.cell_height = MAZE_HEIGHT//ROWS
        self.walls = []
        self.coins = []
        self.enemies = []
        self.e_pos = []
        self.p_pos = None

        self.load()

        self.player = Player(self, copy.copy(self.p_pos))
        self.make_enemies()

    def run(self):
        while self.running:
            if self.state == 'start':
                self.start_events()
                self.start_draw()

            elif self.state == 'playing':
                self.playing_events()
                self.playing_update() 
                self.playing_draw()

            else:
                self.running = False

            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()

################ HELPER FUNCTIONS ################

    def draw_text(self, words, screen, pos, size, color, font_name, centered = False):
        font = pygame.font.SysFont(font_name, size)
        text = font.render(words, False, color)
        text_size = text.get_size()
        if centered:
            pos[0] = pos[0]-text_size[0]//2
            pos[1] = pos[1]-text_size[1]//2

        screen.blit(text, pos)

    def load(self):
        self.background = pygame.image.load('maze.png')
        self.background = pygame.transform.scale(self.background, (MAZE_WIDTH, MAZE_HEIGHT))
        
        # Opening walls file
        # Creating walls list with coords of walls
        with open('walls.txt', 'r') as file:
            for yidx, line in enumerate(file):
                for xidx, char in enumerate(line):
                    if char == "1":
                        self.walls.append(vec(xidx,yidx))
                    elif char == 'C':
                        self.coins.append(vec(xidx,yidx))
                    elif char == 'P':
                        self.p_pos = [xidx,yidx]
                    elif char in ['2','3','4','5']:
                        self.e_pos.append([xidx,yidx])
                    elif char == 'B':
                        pygame.draw.rect(self.background, BLACK, (xidx*self.cell_width, yidx*self.cell_height, self.cell_width, self.cell_height))

    def make_enemies(self):
        for idx, pos in enumerate(self.e_pos):
            self.enemies.append(Enemy(self, vec(pos), idx))

    def draw_grid(self):
        for x in range(WIDTH//self.cell_width):
            pygame.draw.line(self.background, GRAY, (x*self.cell_width,0), (x*self.cell_width, HEIGHT))
        
        for y in range(HEIGHT//self.cell_height):
            pygame.draw.line(self.background, GRAY, (0,y*self.cell_height), (WIDTH, y*self.cell_height))

        #for elem in self.coins:
        #    pygame.draw.circle(self.background, BLUE, (int(elem.x*self.cell_width)+self.cell_width//2, int(elem.y*self.cell_height+self.cell_height//2)), self.cell_height//4)

################ INTRO FUNCTIONS ################

    def start_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.state = 'playing'

    def start_draw(self):
        self.screen.fill(BLACK)
        self.draw_text('PAC MAN', self.screen, [WIDTH//2, HEIGHT//2-150], 80, YELLOW, 'pacfont', True)
        self.draw_text('PUSH SPACE BAR TO PLAY', self.screen, [WIDTH//2,HEIGHT//2], START_TEXT_SIZE, (170,132,58), START_FONT, True)
        self.draw_text('1 PLAYER ONLY', self.screen, [WIDTH//2,HEIGHT//2+50], START_TEXT_SIZE, (44, 167, 198), START_FONT, True)
        
        # UPPER HUD
        self.draw_text('HIGH SCORE', self.screen, [WIDTH//2,10], START_TEXT_SIZE, (255,255,255), START_FONT, True)
        self.draw_text('000000', self.screen, [WIDTH//2,10 + START_TEXT_SIZE], START_TEXT_SIZE, (255,255,255), START_FONT, True)
        
        pygame.display.update()

################ PLAYING FUNCTIONS ################

    def playing_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.player.move(vec(-1,0))
                if event.key == pygame.K_RIGHT:
                    self.player.move(vec(1,0))
                if event.key == pygame.K_UP:
                    self.player.move(vec(0,-1))
                if event.key == pygame.K_DOWN:
                    self.player.move(vec(0,1))

    def playing_update(self):
        self.player.update()
        for enemy in self.enemies:
            enemy.update()

        for enemy in self.enemies:
            if enemy.grid_pos == self.player.grid_pos:
                self.remove_life()

    def playing_draw(self):
        self.screen.fill(BLACK)
        self.screen.blit(self.background, (T_B_BUFFER//2, T_B_BUFFER//2))
        self.draw_coins()
        #self.draw_grid()

        # UPPER HUD
        self.draw_text('HIGH SCORE: 0', self.screen, [WIDTH//2,10], START_TEXT_SIZE, (255,255,255), START_FONT, True)
        self.draw_text(f'SCORE: {self.player.score}', self.screen, [4,2], START_TEXT_SIZE, (255,255,255), START_FONT)

        self.player.draw()
        for enemy in self.enemies:
            enemy.draw()
        pygame.display.update()

    def remove_life(self):
        self.player.lives -= 1
        if self.player.lives == 0:
            self.state = 'game over'
        else:
            self.player.grid_pos = self.p_pos
            self.player.pix_pos = self.player.get_pix_pos()
            self.player.direction = vec(1,0)

            
    def draw_coins(self):
        for coin in self.coins:
            pygame.draw.circle(self.screen, GOLD, (int(coin.x*self.cell_width)+self.cell_width//2+T_B_BUFFER//2, int(coin.y*self.cell_height)+self.cell_height//2+T_B_BUFFER//2), 5)

################ GAME OVER FUNCTIONS ################


