class TicTacToe:
    def __init__(self):
       self.board = [" " for _ in range(9)]
       self.human_player = "O"
       self.ai_player = "X"

    def print_board(self):
       """Print the current state of the board"""
       for i in range(0, 9, 3):
           print(f"{self.board[i]} | {self.board[i+1]} | {self.board[i+2]}")
           if i < 6:
               print("---------")

    def available_moves(self):
       """Returns list of available moves (indices of empty squares)"""
       return [i for i, spot in enumerate(self.board) if spot == " "]
   

    def make_move(self, position, player):
       """Make a move on the board"""
       if self.board[position] == " ":
           self.board[position] = player
           return True
       return False
    
    def is_board_full(self):
       """Check if the board is full"""
       return " " not in self.board
    

    def check_winner(self):
       """Check if there's a winner. Returns winner symbol or None"""
       for i in range(0, 9, 3):
           if self.board[i] == self.board[i + 1] == self.board[i + 2] != " ":
               return self.board[i]

       for i in range(3):
           if self.board[i] == self.board[i + 3] == self.board[i + 6] != " ":
               return self.board[i]

       if self.board[0] == self.board[4] == self.board[8] != " ":
           return self.board[0]
       if self.board[2] == self.board[4] == self.board[6] != " ":
           return self.board[2]

       return None
    

    def game_over(self):
       """Check if the game is over"""
       return self.check_winner() is not None or self.is_board_full()
    
    def minimax(self, depth, is_maximizing):
        winner = self.check_winner()
        if winner == self.ai_player:
            return 1
        if winner == self.human_player:
            return -1
        if self.is_board_full():
            return 0

        if is_maximizing:
            best_score = float("-inf")
            for move in self.available_moves():
                self.board[move] = self.ai_player
                score = self.minimax(depth + 1, False)
                self.board[move] = " "
                best_score = max(score, best_score)
            return best_score

        else:
            best_score = float("inf")
            for move in self.available_moves():
                self.board[move] = self.human_player
                score = self.minimax(depth + 1, True)
                self.board[move] = " "
                best_score = min(score, best_score)
            return best_score
       
       
    def get_best_move(self):
        best_score = float("-inf")
        best_move = None

        for move in self.available_moves():
            self.board[move] = self.ai_player
            score = self.minimax(0, False)
            self.board[move] = " "

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    

    def play_game(self):
       """Main game loop"""
       print("Welcome to Tic Tac Toe!")
       print("You are 'O' and the AI is 'X'")
       print("Enter positions (0-8) as shown below:")
       print("0 | 1 | 2")
       print("---------")
       print("3 | 4 | 5")
       print("---------")
       print("6 | 7 | 8")
       print("\n")
       import random

       ai_turn = random.choice([True, False])

       while not self.game_over():
            self.print_board()

            if ai_turn:
                print("\nAI's turn...")
                move = self.get_best_move()
                self.make_move(move, self.ai_player)
            else:
                while True:
                    try:
                        move = int(input("\nYour turn (0-8): "))
                        if 0 <= move <= 8 and self.make_move(move, self.human_player):
                            break
                        else:
                            print("Invalid move! Try again.")
                    except ValueError:
                        print("Please enter a number between 0 and 8!")

            ai_turn = not ai_turn


       self.print_board()
       winner = self.check_winner()
       if winner == self.ai_player:
            print("\nAI wins!")
       elif winner == self.human_player:
            print("\nCongratulations! You win!")
       else:
            print("\nIt's a tie!")


if __name__ == "__main__":
   game = TicTacToe()
   game.play_game()