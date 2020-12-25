import numpy as np
import random
from . import pieces

board = np.ones((10, 10))

boardWidth = 10
boardHeigh = 10

blank = '.'
full = 'O'

piece_list = {'A': pieces.A,
          'B': pieces.B,
          'C': pieces.C,
          'D': pieces.D,
          'E': pieces.E,
          'F': pieces.F,
          'G': pieces.G}

color_board = 8
colors = {'A': 1,
          'B': 2,
          'C': 3,
          'D': 4,
          'E': 5,
          'F': 6,
          'G': 7}


class GameState:
    def __init__(self):
        self.board = self.getBlankBoard()
        self.currentPiece = None
        self.str_board = []
        self.step = 0
        self.score = 0

    def getActionSet(self):
        return range(boardWidth * boardHeigh)

    def getBlankBoard(self):
        self.board = []
        for i in range(boardWidth):
            self.board = np.append(self.board, [blank] * boardWidth)
        self.board = np.reshape(self.board, (boardWidth, boardHeigh))
        return self.board

    def getReward(self, valid=True):
        counter_ones = 0
        for row in self.int_board:
            for value in row:
                if value != color_board:
                    counter_ones += 1
        if valid:
            self.score = self.score + 10  # useful to display results
            reward = counter_ones ** 2 / 100 + 100
        else:
            reward = counter_ones ** 2 / 100 * 0.01
        return reward

    def predictedPiece(self, shape, x, y):
        newPiece = {'shape': shape,
                    'rotation': random.randint(0, len(piece_list[shape]) - 1),
                    'x': int(x),
                    'y': int(y),  # start it above the self.board (i.e. less than 0)
                    'color': colors[shape]}
        return newPiece

    def isValidPosition(self):
        valid = True
        for row in range(boardWidth):
            for column in range(boardHeigh):
                if piece_list[self.currentPiece['shape']][self.currentPiece['rotation']][row][column] == full:
                    if self.currentPiece['x'] + row > boardWidth - 1 or \
                            self.currentPiece['y'] + column > boardHeigh - 1:
                        valid = False
                    else:
                        if valid:
                            if self.int_board[self.currentPiece['x'] + row][self.currentPiece['y'] + column] == color_board:
                                valid = True
                            else:
                                valid = False
        return valid

    def addToBoard(self, shape):
        if self.isValidPosition() == True:
            for row in range(boardWidth):
                for column in range(boardHeigh):
                    if piece_list[self.currentPiece['shape']][self.currentPiece['rotation']][row][column] != blank:
                        try:
                            self.board[row + self.currentPiece['x']][column + self.currentPiece['y']] = \
                            self.currentPiece['color']
                        except:
                            pass
            return self.getReward()
        else:
            return self.getReward(valid=False)

    def convertToIntBoard(self):
        self.int_board = []
        for row in range(boardWidth):
            for column in range(boardHeigh):
                if self.board[row, column] == blank:
                    self.int_board = np.append(self.int_board, int(color_board))
                else:
                    self.int_board = np.append(self.int_board, int(self.board[row, column]))

        self.int_board = np.reshape(self.int_board, (boardWidth, boardHeigh))
        return self.int_board

    def convertToIntPiece(self, board):
        int_board = []
        for row in board:
            for column in row:
                if column == blank:
                    int_board = np.append(int_board, color_board)
                else:
                    int_board = np.append(int_board, self.currentPiece['color'])

        int_board = np.reshape(int_board, (boardWidth, boardHeigh))
        return int_board

    def convertPieceToNumpy(self):
        piece_np = []
        if self.step < 6:
            for row in piece_list[list(piece_list)[self.step + 1]][0]:
                for column in row:
                    piece_np = np.append(piece_np, column)
            piece_np = np.reshape(piece_np, (boardWidth, boardHeigh))
        else:
            for row in piece_list[list(piece_list)[0]][0]:
                for column in row:
                    piece_np = np.append(piece_np, column)
            piece_np = np.reshape(piece_np, (boardWidth, boardHeigh))
        return piece_np

    def boardToVector(self, board, piece):
        board = board.flatten()
        board = np.reshape(board, (1, boardWidth * boardHeigh))
        piece = piece.flatten()
        piece = np.reshape(piece, (1, boardWidth * boardHeigh))
        state = np.concatenate((board, piece), axis=1)
        return state

    def frame_step(self, action):
        done = False
        if action == 100:  # Restart-->Empty board and piece
            self.step = 0
            board = self.getBlankBoard()
            self.convertToIntBoard()
            board = self.int_board
            state = self.boardToVector(board, board)
            done = True
            return state

        else:
            if self.step == 6:
                done = True

            if len(str(action)) == 1:
                x = 0
                y = str(action)[0]
            else:
                x = str(action)[0]
                y = str(action)[1]
            self.currentPiece = self.predictedPiece(list(piece_list)[self.step], x, y)

            reward = self.addToBoard(list(piece_list)[self.step])
            self.convertToIntBoard()
            board = self.int_board

            piece_np = self.convertPieceToNumpy()
            piece = self.convertToIntPiece(piece_np)

            state = self.boardToVector(board, piece)

            self.step += 1
            return state, reward, done

# if __name__ == "__main__":
#    print('inici')
#    gamestate =  GameState()
#    gamestate.frame_step(1)
#    print(gamestate.getReward())
