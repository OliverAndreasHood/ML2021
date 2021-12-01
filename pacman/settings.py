from pygame.math import Vector2 as vec

# screen settings
WIDTH, HEIGHT = 610, 670
ROWS = 30
COLS = 28
T_B_BUFFER = 50
MAZE_WIDTH, MAZE_HEIGHT = WIDTH-T_B_BUFFER, HEIGHT-T_B_BUFFER
FPS = 60

# colour settings
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
YELLOW = (255,255,0)
GOLD = (212,175,55)
GRAY = (107,107,107)
CYAN = (0,255,255)
VIOLET = (127,0,255)


# font settings
START_TEXT_SIZE = 16
START_FONT = 'arcade normal'

# player settings
#PLAYER_START_POS = vec(1,1)
PLAYER_COLOR = YELLOW

# mob settings
E_COLORS = [RED, GREEN, CYAN, VIOLET]