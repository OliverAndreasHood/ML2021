from settings import *
import pygame
vec = pygame.math.Vector2

class Player:
    def __init__(self, app, pos):
        self.app = app
        # Stats
        self.grid_pos = pos
        self.pix_pos = self.get_pix_pos()
        self.score = 0
        self.speed = 2

        # Movement        
        self.direction = vec(1,0) 
        self.stored_direction = None
        self.able_to_move = True

    def update(self):
        if self.able_to_move:
            self.pix_pos += self.direction*self.speed
        
        if self.time_to_move(): 
            if self.stored_direction != None:
                # Debug
                print(self.stored_direction, self.grid_pos)
                
                self.direction = self.stored_direction     
                self.stored_direction = None
            
            self.able_to_move = self.can_move()

        # setting grid position in reference to pix position
        self.grid_pos[0] = (self.pix_pos[0]-T_B_BUFFER + self.app.cell_width//2)//self.app.cell_width+1
        self.grid_pos[1] = (self.pix_pos[1]-T_B_BUFFER + self.app.cell_height//2)//self.app.cell_height+1

        if self.on_coin():
            self.eat_coin()

    def draw(self):
        pygame.draw.circle(self.app.screen, PLAYER_COLOR, (int(self.pix_pos.x),int(self.pix_pos.y)), self.app.cell_width//2-2)

        # drawing the grid position rectangle
        #pygame.draw.rect(self.app.screen, RED, (self.grid_pos[0]*self.app.cell_width+T_B_BUFFER//2, self.grid_pos[1]*self.app.cell_height+T_B_BUFFER//2, self.app.cell_width, self.app.cell_height), 1)

    def get_pix_pos(self):
        return vec((self.grid_pos.x*self.app.cell_width)+T_B_BUFFER//2+self.app.cell_width//2, (self.grid_pos.y*self.app.cell_height)+T_B_BUFFER//2+self.app.cell_height//2)

################ MOVEMENT ################

    def move(self, direction):
        self.stored_direction = direction

    def time_to_move(self):
        if (self.pix_pos.x+T_B_BUFFER//2) % self.app.cell_width == 0:
            if self.direction == vec(1,0) or self.direction == vec(-1,0):
                return True

        if (self.pix_pos.y+T_B_BUFFER//2) % self.app.cell_height == 0:
            if self.direction == vec(0,1) or self.direction == vec(0,-1):
                return True

    def can_move(self):
        for wall in self.app.walls:
            if vec(self.grid_pos+self.direction) == wall:
                return False
        return True

################ COINS ################

    def on_coin(self):
        if self.grid_pos in self.app.coins:
            if (self.pix_pos.x+T_B_BUFFER//2) % self.app.cell_width == 0:
                if self.direction == vec(1,0) or self.direction == vec(-1,0):
                    return True

            if (self.pix_pos.y+T_B_BUFFER//2) % self.app.cell_height == 0:
                if self.direction == vec(0,1) or self.direction == vec(0,-1):
                    return True
        return False

    def eat_coin(self):
        self.app.coins.remove(self.grid_pos)
        self.score += 1
