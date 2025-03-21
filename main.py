import pygame
import sys
import random

# Константы
WIDTH, HEIGHT = 500, 500
CELL_SIZE = WIDTH // 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
GRAY = (169, 169, 169)
GREEN = (0, 128, 0)
FPS = 30

class Fanorona:
    def __init__(self):
        self.board = [
            [-1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
            [-1, 1, 0, -1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ]

    def get_possible_directions(self, x, y):
        if (x % 2 == 0 and y % 2 == 0) or (x % 2 == 1 and y % 2 == 1):
            return [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        else:
            return [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def get_possible_moves(self, player):
        player_chips = [(i, j) for i in range(5) for j in range(5) if self.board[i][j] == player]
        opponent = 1 if player == -1 else -1
        moves = {}
        moves_n = {}

        for x, y in player_chips:
            directions = self.get_possible_directions(x, y)
            for direction in directions:
                dx, dy = direction
                nx, ny = x + dx, y + dy

                if nx < 0 or nx > 4 or ny < 0 or ny > 4 or self.board[nx][ny] != 0:
                    continue

                if (x, y) not in moves_n:
                    moves_n[(x, y)] = []
                moves_n[(x, y)].append(direction)

                nnx, nny = x - dx, y - dy
                if 0 <= nnx < 5 and 0 <= nny < 5 and self.board[nnx][nny] == opponent:
                    if (x, y) not in moves:
                        moves[(x, y)] = []
                    moves[(x, y)].append(direction)
                    continue

                nnx, nny = nx + dx, ny + dy
                if 0 <= nnx < 5 and 0 <= nny < 5 and self.board[nnx][nny] == opponent:
                    if (x, y) not in moves:
                        moves[(x, y)] = []
                    moves[(x, y)].append(direction)

        return (bool(moves), moves) if moves else (False, moves_n)

    def get_possible_moves_chip(self, player, x, y, directions_before=None, player_moves=None):
        if player_moves is None:
            player_moves = []
        if directions_before is None:
            directions_before = []

        opponent = 1 if player == -1 else -1
        directions = self.get_possible_directions(x, y)
        moves = {}

        for direction in directions:
            dx, dy = direction
            nx, ny = x + dx, y + dy
            if direction in directions_before or (nx, ny) in player_moves or nx < 0 or nx > 4 or ny < 0 or ny > 4 or self.board[nx][ny] != 0:
                continue

            nnx, nny = x - dx, y - dy
            if 0 <= nnx < 5 and 0 <= nny < 5 and self.board[nnx][nny] == opponent:
                if (x, y) not in moves:
                    moves[(x, y)] = []
                moves[(x, y)].append(direction)
                continue

            nnx, nny = nx + dx, ny + dy
            if 0 <= nnx < 5 and 0 <= nny < 5 and self.board[nnx][nny] == opponent:
                if (x, y) not in moves:
                    moves[(x, y)] = []
                moves[(x, y)].append(direction)

        return moves

    def check_move(self, player, x, y, direction):
        dx, dy = direction
        opponent = 1 if player == -1 else -1

        def count_opponent_chips(nnx, nny, dx, dy):
            count = 0
            while 0 <= nnx < 5 and 0 <= nny < 5:
                if self.board[nnx][nny] == opponent:
                    count += 1
                nnx, nny = nnx + dx, nny + dy
            return count

        off_c = count_opponent_chips(x + 2 * dx, y + 2 * dy, dx, dy)
        def_c = count_opponent_chips(x - dx, y - dy, -dx, -dy)

        return off_c, def_c

    def make_move(self, player, x, y, direction, offence=False, kill=True):
        dx, dy = direction
        nx, ny = x + dx, y + dy
        self.board[nx][ny] = player
        self.board[x][y] = 0
        opponent = 1 if player == -1 else -1
        if kill:
            nnx, nny = (x + 2 * dx, y + 2 * dy) if offence else (x - dx, y - dy)
            while 0 <= nnx < 5 and 0 <= nny < 5:
                if self.board[nnx][nny] != opponent:
                    return
                self.board[nnx][nny] = 0
                nnx, nny = (nnx + dx, nny + dy) if offence else (nnx - dx, nny - dy)

class RandomAgent:
    def select_move(self, possible_moves):
        move_from = random.choice(list(possible_moves.keys()))
        direction = random.choice(possible_moves[move_from])
        return move_from, direction

class FanoronaGame:
    def __init__(self, fanorona):
        self.fanorona = fanorona
        self.selected_chip = None
        self.possible_moves = {}
        self.player = 1

    def draw_board(self, screen):
        screen.fill(GREEN)
        
        for x in range(5):
            for y in range(5):
                if x < 4:
                    pygame.draw.line(screen, BLACK, (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), ((x + 1) * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), 1)
                if y < 4:
                    pygame.draw.line(screen, BLACK, (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), (x * CELL_SIZE + CELL_SIZE // 2, (y + 1) * CELL_SIZE + CELL_SIZE // 2), 1)
                if (x + y) % 2 == 0 and x < 4 and y < 4:
                    #вниз вправо
                    pygame.draw.line(screen, BLACK, (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), ((x + 1) * CELL_SIZE + CELL_SIZE // 2, (y + 1) * CELL_SIZE + CELL_SIZE // 2), 1)
                if (x + y) % 2 == 0 and x > 0 and y < 4:
                    # вниз влево
                    pygame.draw.line(screen, BLACK, (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), ((x - 1) * CELL_SIZE + CELL_SIZE // 2, (y + 1) * CELL_SIZE + CELL_SIZE // 2), 1)

        for x in range(5):
            for y in range(5):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)

                chip = self.fanorona.board[y][x]
                if chip == 1:
                    pygame.draw.circle(screen, WHITE, rect.center, CELL_SIZE // 3)
                elif chip == -1:
                    pygame.draw.circle(screen, BLACK, rect.center, CELL_SIZE // 3)


        if self.selected_chip:
            x, y = self.selected_chip
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, YELLOW, rect, 3)

            for direction in self.possible_moves.get(self.selected_chip, []):
                dx, dy = direction
                px, py = x + dx, y + dy
                if 0 <= px < 5 and 0 <= py < 5:
                    center = (px * CELL_SIZE + CELL_SIZE // 2, py * CELL_SIZE + CELL_SIZE // 2)
                    pygame.draw.circle(screen, YELLOW, center, 10)

    def handle_click(self, pos):
        x, y = pos[0] // CELL_SIZE, pos[1] // CELL_SIZE

        if (x, y) == self.selected_chip:
            self.selected_chip = None
            self.possible_moves = {}
            return

        if self.fanorona.board[y][x] == self.player:
            self.selected_chip = (x, y)
            _, self.possible_moves = self.fanorona.get_possible_moves(self.player)

        elif self.selected_chip and (x, y) in [(self.selected_chip[0] + dx, self.selected_chip[1] + dy) 
                                                for dx, dy in self.possible_moves.get(self.selected_chip, [])]:
            direction = (x - self.selected_chip[0], y - self.selected_chip[1])
            self.fanorona.make_move(self.player, self.selected_chip[0], self.selected_chip[1], direction)
            self.player = -self.player
            self.selected_chip = None
            self.possible_moves = {}

def play_game():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Fanorona")
    clock = pygame.time.Clock()

    fanorona = Fanorona()
    game = FanoronaGame(fanorona)
    random_agent = RandomAgent()

    running = True
    player = 1

    while running:
        if player == 1:
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    game.handle_click(pygame.mouse.get_pos())
                    if game.selected_chip is None:
                        player = -player
        else:
            canKill, possible_moves = fanorona.get_possible_moves(player)
            if not possible_moves:
                running = False
                break

            if canKill:
                move = random_agent.select_move(possible_moves)
                position, direction = move
                x, y = position
                off_c, def_c = fanorona.check_move(player, x, y, direction)
                fanorona.make_move(player, x, y, direction, offence=off_c > def_c)
                positions = [position]
                directions = [direction]
                x, y = x + direction[0], y + direction[1]
                while True:
                    possible_chip_moves = fanorona.get_possible_moves_chip(player, x, y, directions_before=directions, player_moves=positions)
                    if not possible_chip_moves:
                        break
                    move = random_agent.select_move(possible_chip_moves)
                    position, direction = move
                    x, y = position
                    positions.append(position)
                    directions.append(direction)
                    off_c, def_c = fanorona.check_move(player, x, y, direction)
                    fanorona.make_move(player, x, y, direction, offence=off_c > def_c)
                    x, y = x + direction[0], y + direction[1]
            else:
                move = random_agent.select_move(possible_moves)
                position, direction = move
                x, y = position
                fanorona.make_move(player, x, y, direction, kill=False)
            player = -player

        game.draw_board(screen)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    play_game()