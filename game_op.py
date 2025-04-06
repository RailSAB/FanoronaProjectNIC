import pygame
import random
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
import os
import glob
import copy
import csv
import sys


pygame.init()
WIDTH, HEIGHT = 600, 700
ROWS, COLS = 5, 5
NODE_SIZE = 80
MARGIN = 50
RADIUS = 20
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Fanorona")
FONT = pygame.font.SysFont("Arial", 28)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
YELLOW = (255, 255, 0)
BLUE = (50, 50, 255)

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
            while 0 <= nnx < 5 and 0 <= nny < 5 and self.board[nnx][nny] == opponent:
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
    
    def check_game_over(self):
        white_pieces = sum(cell == 1 for row in self.board for cell in row)
        black_pieces = sum(cell == -1 for row in self.board for cell in row)

        if white_pieces == 0:
            return True, -1  
        elif black_pieces == 0:
            return True, 1  

        white_can_move = self.get_possible_moves(1)[1]
        black_can_move = self.get_possible_moves(-1)[1]

        if not white_can_move and not black_can_move:
            return True, 0  
        elif not white_can_move:
            return True, -1  
        elif not black_can_move:
            return True, 1  

        return False, None


class RandomAgent:
    def select_move(self, moves):
        src = random.choice(list(moves.keys()))
        dir = random.choice(moves[src])
        return src, dir
    
def create_model(input_shape, output_size):
        model = Sequential([
            Flatten(input_shape=input_shape),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(output_size, activation='softmax')
        ])
        return model

def load_weights(model):
    weights_path = "weights/best_model/best_model_weights.weights.h5"
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
    else:
        print(f"Warning: Weight file {weights_path} not found.")

class NeuralNetworkAgent:
    def __init__(self, model):
        self.model = model

    def predict_move(self, board, possible_moves):
        input_board = np.array(board).flatten().reshape(1, -1)
        probabilities = self.model.predict(input_board)[0]

        valid_moves = [
            (probabilities[x * 5 * 8 + y * 8 + self.direction_to_index(direction)], (x, y), direction)
            for (x, y), directions in possible_moves.items() for direction in directions
        ]

        valid_moves.sort(reverse=True, key=lambda x: x[0])
        return valid_moves[0][1], valid_moves[0][2]

    @staticmethod
    def direction_to_index(direction):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        return directions.index(direction)


def get_node_position(i, j):
    x = MARGIN + j * NODE_SIZE + NODE_SIZE // 2
    y = MARGIN + i * NODE_SIZE + NODE_SIZE // 2 + 100
    return x, y

def draw_board(game, selected=None, highlight_moves=None, log_text=""):
    SCREEN.fill((200, 200, 200))
    
    text_surface = FONT.render(log_text, True, BLACK)
    SCREEN.blit(text_surface, (WIDTH // 2 - text_surface.get_width() // 2, 20))

    for i in range(ROWS):
        for j in range(COLS):
            x, y = get_node_position(i, j)
            for dx, dy in game.get_possible_directions(i, j):
                ni, nj = i + dx, j + dy
                if 0 <= ni < ROWS and 0 <= nj < COLS:
                    nx, ny = get_node_position(ni, nj)
                    pygame.draw.line(SCREEN, BLACK, (x, y), (nx, ny), 2)

    for i in range(ROWS):
        for j in range(COLS):
            x, y = get_node_position(i, j)
            piece = game.board[i][j]
            if piece == 1:
                pygame.draw.circle(SCREEN, WHITE, (x, y), RADIUS)
            elif piece == -1:
                pygame.draw.circle(SCREEN, BLACK, (x, y), RADIUS)

    if selected:
        x, y = get_node_position(*selected)
        pygame.draw.rect(SCREEN, YELLOW, (x - RADIUS - 4, y - RADIUS - 4, 2 * RADIUS + 8, 2 * RADIUS + 8), 3)

    if highlight_moves:
        for dx, dy in highlight_moves:
            nx, ny = selected[0] + dx, selected[1] + dy
            if 0 <= nx < ROWS and 0 <= ny < COLS:
                cx, cy = get_node_position(nx, ny)
                pygame.draw.circle(SCREEN, YELLOW, (cx, cy), 6)

    pygame.display.flip()

def draw_offence_defence_choice():
    
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(200)
    overlay.fill((50, 50, 50))
    SCREEN.blit(overlay, (0, 0))

    
    title = FONT.render("Choose Capture Type", True, WHITE)
    SCREEN.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 2 - 100))

    
    offence_rect = pygame.Rect(WIDTH // 2 - 120, HEIGHT // 2 - 30, 100, 50)
    defence_rect = pygame.Rect(WIDTH // 2 + 20, HEIGHT // 2 - 30, 100, 50)
    pygame.draw.rect(SCREEN, (200, 0, 0), offence_rect)
    pygame.draw.rect(SCREEN, (0, 0, 200), defence_rect)

    off_text = FONT.render("Offence", True, WHITE)
    def_text = FONT.render("Defence", True, WHITE)
    SCREEN.blit(off_text, (offence_rect.x + 5, offence_rect.y + 10))
    SCREEN.blit(def_text, (defence_rect.x + 5, defence_rect.y + 10))

    pygame.display.flip()
    return offence_rect, defence_rect

def main():
    clock = pygame.time.Clock()
    game = Fanorona()
    # agent = RandomAgent()
    model = create_model((25,), 200)
    load_weights(model)
    agent = NeuralNetworkAgent(model)
    running = True
    selected = None
    player_turn = 1
    chain_mode = False
    chain_positions = []
    chain_directions = []

    while running:
        clock.tick(60)

        if player_turn == 1:
            draw_board(game, log_text="Agent is moving...")
            pygame.display.flip()
            time.sleep(1)
            canKill, possible_moves = game.get_possible_moves(player_turn)
            if canKill:
                # src, direction = agent.select_move(possible_moves)
                src, direction = agent.predict_move(game.board, possible_moves)
                x, y = src
                off_c, def_c = game.check_move(1, x, y, direction)
                game.make_move(1, x, y, direction, offence=off_c > def_c)
                positions = [src]
                directions = [direction]
                x, y = x + direction[0], y + direction[1]
                while True:
                    possible_chip_moves = game.get_possible_moves_chip(1, x, y, directions_before=directions,
                                                                       player_moves=positions)
                    if not possible_chip_moves:
                        break
                    draw_board(game, log_text="Agent is moving...")
                    pygame.display.flip()
                    time.sleep(2)
                    # src, direction = agent.select_move(possible_chip_moves)
                    src, direction = agent.predict_move(game.board, possible_chip_moves)
                    x, y = src
                    off_c, def_c = game.check_move(1, x, y, direction)
                    game.make_move(1, x, y, direction, offence=off_c > def_c)
                    positions.append(src)
                    directions.append(direction)
                    x, y = x + direction[0], y + direction[1]
            else:
                # src, direction = agent.select_move(possible_moves)
                src, direction = agent.predict_move(game.board, possible_moves)
                x, y = src
                game.make_move(1, x, y, direction, kill=False)
            draw_board(game, log_text="Agent moved.")
            pygame.display.flip()
            time.sleep(2)
            player_turn = -1
            game_over, winner = game.check_game_over()
            if game_over:
                draw_board(game, log_text="Game Over: " + (
                    "Draw" if winner == 0 else ("White wins!" if winner == 1 else "Black wins!")))
                pygame.display.flip()
                time.sleep(5)
                running = False
            continue

        if not chain_mode:
            canKill, possible_moves = game.get_possible_moves(player_turn)
        else:
            possible_moves = game.get_possible_moves_chip(player_turn, selected[0], selected[1],
                                                          directions_before=chain_directions,
                                                          player_moves=chain_positions)
            canKill = True

        draw_board(game, selected, possible_moves.get(selected, []) if selected else None, "Your move")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                for i in range(ROWS):
                    for j in range(COLS):
                        x, y = get_node_position(i, j)
                        if (x - mx) ** 2 + (y - my) ** 2 <= RADIUS ** 2:
                            if selected is None:
                                if (i, j) in possible_moves:
                                    selected = (i, j)
                            else:
                                dx, dy = i - selected[0], j - selected[1]
                                if (dx, dy) in possible_moves[selected]:
                                    off_c, def_c = game.check_move(-1, selected[0], selected[1], (dx, dy))
                                    offence = True
                                    if off_c > 0 and def_c > 0:
                                        offence_rect, defence_rect = draw_offence_defence_choice()
                                        choosing = True
                                        while choosing:
                                            for event in pygame.event.get():
                                                if event.type == pygame.QUIT:
                                                    pygame.quit()
                                                    sys.exit()
                                                elif event.type == pygame.MOUSEBUTTONDOWN:
                                                    mx, my = pygame.mouse.get_pos()
                                                    if offence_rect.collidepoint(mx, my):
                                                        offence = True
                                                        choosing = False
                                                    elif defence_rect.collidepoint(mx, my):
                                                        offence = False
                                                        choosing = False

                                    else:
                                        offence = off_c > def_c

                                    game.make_move(-1, selected[0], selected[1], (dx, dy), offence=offence)
                                    next_x, next_y = selected[0] + dx, selected[1] + dy
                                    if not chain_mode:
                                        chain_positions = [selected]
                                        chain_directions = [(dx, dy)]
                                    else:
                                        chain_positions.append(selected)
                                        chain_directions.append((dx, dy))
                                    selected = (next_x, next_y)
                                    if game.get_possible_moves_chip(-1, next_x, next_y,
                                                                    directions_before=chain_directions,
                                                                    player_moves=chain_positions) and canKill:
                                        chain_mode = True
                                    else:
                                        chain_mode = False
                                        selected = None
                                        player_turn = 1
                                        game_over, winner = game.check_game_over()
                                        if game_over:
                                            draw_board(game, log_text="Game Over: " + ("Draw" if winner == 0 else (
                                                "White wins!" if winner == 1 else "Black wins!")))
                                            pygame.display.flip()
                                            time.sleep(5)
                                            running = False
                                else:
                                    selected = None
                                    if not chain_mode:
                                        chain_positions = []
                                        chain_directions = []
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()