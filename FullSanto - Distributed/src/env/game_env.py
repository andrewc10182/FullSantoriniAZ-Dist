import enum
import numpy as np
import random
Winner = enum.Enum("Winner", "black white draw")
Player = enum.Enum("Player", "black white")

class GameEnv:
    def __init__(self):
        self.board = None
        self.turn = 0
        self.done = False
        self.winner = None  # type: Winner
        self.resigned = False

    def reset(self):
        self.board = []
        for i in range(5):
            self.board.append([])
            for j in range(5):
                self.board[i].append(' ')
        self.turn = 0
        self.done = False
        self.winner = None
        self.resigned = False
        return self

    def update(self, board):
        self.board = np.copy(board)
        self.turn = self.turn_n()
        self.done = False
        self.winner = None
        self.resigned = False
        return self

    def turn_n(self):
        turn = 0
        for i in range(5):
            for j in range(5):
                if self.board[i][j]=='1' or self.board[i][j] == 'B' or self.board[i][j] == 'X':
                    turn += 1
                if self.board[i][j]=='2' or self.board[i][j] == 'C' or self.board[i][j] == 'Y':
                    turn += 2
                if self.board[i][j]=='3' or self.board[i][j] == 'D' or self.board[i][j] == 'Z':
                    turn += 3
                if self.board[i][j]=='4':
                    turn += 4
        return turn

    def player_turn(self):
        if self.turn % 2 == 0:
            return Player.white
        else:
            return Player.black

    def step(self, action):
        if sum(self.legal_moves())==0 and action!=999:
            self._resigned()
            return self.board, {}
        if action==999: # A setting move at the beginning of game
            if(any('A' in sublist for sublist in self.board)==False): # Cannot find any A
                self.board[random.randint(0,4)][random.randint(0,4)] = 'A'
                while True:
                    tempi = random.randint(0,4)
                    tempj = random.randint(0,4)
                    if self.board[tempi][tempj] == ' ':
                        self.board[tempi][tempj] = 'A'
                        break
            else: # Cannot find any W
                while True:
                    tempi = random.randint(0,4)
                    tempj = random.randint(0,4)
                    if self.board[tempi][tempj] == ' ':
                        self.board[tempi][tempj] = 'W'
                        break
                while True:
                    tempi = random.randint(0,4)
                    tempj = random.randint(0,4)
                    if self.board[tempi][tempj] == ' ':
                        self.board[tempi][tempj] = 'W'
                        break
        else:
            Posi = 0
            Posj = 0
            MoveToi = 0
            MoveToj = 0
            BuildAti = 0
            BuildAtj = 0
            if(self.turn%2==0): FindPlayer = ['A', 'B', 'C']
            else: FindPlayer = ['W', 'X', 'Y']
            #print('Action',action,'Turn:',self.turn,'FindPlayer',FindPlayer)
            
            if(action<64):
                for i in range(5):
                    find = False
                    for j in range(5):
                        if(self.board[i][j] in FindPlayer):
                            Posi = i
                            Posj = j
                            find = True
                            break
                    if find:
                        break

            else:
                action -=64
                for i in range(4,-1,-1):
                    find = False
                    for j in range(4,-1,-1):
                       if(self.board[i][j] in FindPlayer):
                            Posi = i
                            Posj = j
                            find = True
                            break
                    if find:
                        break

            #Delete from Old Position
            self.board[Posi][Posj] = self.Player_MoveAway(self.board[Posi][Posj])

            #Make Move to new Position                    
            MoveCode = int(action/8)+1

            if(MoveCode>4): MoveCode+=1 #Changes 5 to 8 into 6 to 9
            if(MoveCode == 1):
                Posi -= 1
                Posj -= 1
            if(MoveCode == 2):
                Posi -= 1
            if(MoveCode == 3):
                Posi -= 1
                Posj += 1
            if(MoveCode == 4):
                Posj -= 1
            if(MoveCode == 6):
                Posj += 1
            if(MoveCode == 7):
                Posi += 1
                Posj -= 1
            if(MoveCode == 8):
                Posi += 1
            if(MoveCode == 9):
                Posi += 1
                Posj += 1
            self.board[Posi][Posj] = self.Player_MoveTo(self.board[Posi][Posj])

            BuildCode = int(action%8)+1

            if(BuildCode>4): BuildCode+=1

            if(BuildCode == 1): self.board[Posi-1][Posj-1] = self.BuildUpgrade(self.board[Posi-1][Posj-1])
            if(BuildCode == 2): self.board[Posi-1][Posj] = self.BuildUpgrade(self.board[Posi-1][Posj])
            if(BuildCode == 3): self.board[Posi-1][Posj+1] = self.BuildUpgrade(self.board[Posi-1][Posj+1])
            if(BuildCode == 4): self.board[Posi][Posj-1] = self.BuildUpgrade(self.board[Posi][Posj-1])
            if(BuildCode == 6): self.board[Posi][Posj+1] = self.BuildUpgrade(self.board[Posi][Posj+1])
            if(BuildCode == 7): self.board[Posi+1][Posj-1] = self.BuildUpgrade(self.board[Posi+1][Posj-1])
            if(BuildCode == 8): self.board[Posi+1][Posj] = self.BuildUpgrade(self.board[Posi+1][Posj])
            if(BuildCode == 9): self.board[Posi+1][Posj+1] = self.BuildUpgrade(self.board[Posi+1][Posj+1])

        self.turn = self.turn+1
        self.check_for_win()
        return self.board, {}

    def Player_MoveAway(self, Current_Symbol):
        if Current_Symbol == 'A' or Current_Symbol == 'W': return ' '
        if Current_Symbol == 'B' or Current_Symbol == 'X': return '1'
        if Current_Symbol == 'C' or Current_Symbol == 'Y': return '2'

    def Player_MoveTo(self, Current_Level):
        if self.turn%2==0 and Current_Level == ' ': return 'A'
        if self.turn%2==0 and Current_Level == '1': return 'B'
        if self.turn%2==0 and Current_Level == '2': return 'C'
        if self.turn%2==0 and Current_Level == '3': return 'D'
        if self.turn%2==1 and Current_Level == ' ': return 'W'
        if self.turn%2==1 and Current_Level == '1': return 'X'
        if self.turn%2==1 and Current_Level == '2': return 'Y'
        if self.turn%2==1 and Current_Level == '3': return 'Z'

    def BuildUpgrade(self, Current_Level):
        if Current_Level == ' ': return '1'
        if Current_Level == '1': return '2'
        if Current_Level == '2': return '3'
        if Current_Level == '3': return '4'
        

    def Can_Move(self, board, i, j, k):
        # i or j == 4 when the board size is 5x5.  Change to 2 for 3x3
        CurrentRank = 0
        if(board[i][j] == 'B' or board[i][j] == 'X'): CurrentRank = 1
        if(board[i][j] == 'C' or board[i][j] == 'Y'): CurrentRank = 2
        Block = ['A', 'B', 'C', 'D', 'W', 'X', 'Y', 'Z']
        if(k==1 and (i==0 or j==0 or board[i-1][j-1] in Block or (board[i-1][j-1]!=' ' and int(board[i-1][j-1])-1>CurrentRank))):
           return False
        if(k==2 and (i==0 or board[i-1][j] in Block or (board[i-1][j]!=' ' and int(board[i-1][j])-1>CurrentRank))):
           return False
        if(k==3 and (i==0 or j==4 or board[i-1][j+1] in Block or (board[i-1][j+1]!=' ' and int(board[i-1][j+1])-1>CurrentRank))):
           return False
        if(k==4 and (j==0 or board[i][j-1] in Block or (board[i][j-1]!=' ' and int(board[i][j-1])-1>CurrentRank))):
           return False
        if(k==6 and (j==4 or board[i][j+1] in Block or (board[i][j+1]!=' ' and int(board[i][j+1])-1>CurrentRank))):
           return False
        if(k==7 and (i==4 or j==0 or board[i+1][j-1] in Block or (board[i+1][j-1]!=' ' and int(board[i+1][j-1])-1>CurrentRank))):
           return False
        if(k==8 and (i==4 or board[i+1][j] in Block or (board[i+1][j]!=' ' and int(board[i+1][j])-1>CurrentRank))):
           return False
        if(k==9 and (i==4 or j==4 or board[i+1][j+1] in Block or (board[i+1][j+1]!=' ' and int(board[i+1][j+1])-1>CurrentRank))):
           return False
        if(k==5):
           return False
        return True

    def Can_Build(self, board, i, j, k, l): #board, newi, newj, previous k move, new l build
        # i or j == 4 when the board size is 5x5.  Change to 2 for 3x3
        if(k + l == 10): #Building from where it moved from is Valid
           return True    
        if(l==1 and (i==0 or j==0 or (board[i-1][j-1]!=" " and board[i-1][j-1]!='1'and board[i-1][j-1]!='2'and board[i-1][j-1]!='3'))):
           return False
        if(l==2 and (i==0 or (board[i-1][j]!=" "and board[i-1][j]!='1'and board[i-1][j]!='2'and board[i-1][j]!='3'))):
           return False
        if(l==3 and (i==0 or j==4 or (board[i-1][j+1]!=" "and board[i-1][j+1]!='1'and board[i-1][j+1]!='2'and board[i-1][j+1]!='3'))):
           return False
        if(l==4 and (j==0 or (board[i][j-1]!=" "and board[i][j-1]!='1'and board[i][j-1]!='2'and board[i][j-1]!='3'))):
           return False
        if(l==6 and (j==4 or (board[i][j+1]!=" "and board[i][j+1]!='1'and board[i][j+1]!='2'and board[i][j+1]!='3'))):
           return False
        if(l==7 and (i==4 or j==0 or (board[i+1][j-1]!=" "and board[i+1][j-1]!='1'and board[i+1][j-1]!='2'and board[i+1][j-1]!='3'))):
           return False
        if(l==8 and (i==4 or (board[i+1][j]!=" "and board[i+1][j]!='1'and board[i+1][j]!='2'and board[i+1][j]!='3'))):
           return False
        if(l==9 and (i==4 or j==4 or (board[i+1][j+1]!=" "and board[i+1][j+1]!='1'and board[i+1][j+1]!='2'and board[i+1][j+1]!='3'))):
           return False
        if(l==5):
           return False
        return True

    def legal_moves(self):
        #LegalCode = []
        legal = [0] * 128 # Array of Number of Possible Moves

        if(self.turn%2==0): FindPlayer = ['A', 'B', 'C']
        else: FindPlayer = ['W', 'X', 'Y']
        
        WorkerNum=1
        for i in range(5):
            for j in range(5):
                if(self.board[i][j] in FindPlayer):
                   for k in range(1,10): #Move
                       if(self.Can_Move(self.board, i, j, k)):
                           for l in range(1,10): #Move
                               if(self.Can_Move(self.board, i, j, k)):
                                   if(k==1): newi = i-1; newj = j-1
                                   if(k==2): newi = i-1; newj = j
                                   if(k==3): newi = i-1; newj = j+1
                                   if(k==4): newi = i  ; newj = j-1
                                   if(k==6): newi = i  ; newj = j+1
                                   if(k==7): newi = i+1; newj = j-1
                                   if(k==8): newi = i+1; newj = j
                                   if(k==9): newi = i+1; newj = j+1
                                   if(self.Can_Build(self.board, newi, newj, k, l)):
                                       #LegalCode.append(WorkerNum*100+k*10+l)
                                       Posk=k-1
                                       if(Posk>4): Posk-=1
                                       Posl=l-1
                                       if(Posl>4): Posl-=1
                                       legal[(WorkerNum-1)*64 + Posk*8 + Posl] = 1
                   WorkerNum+=1
        return legal

    def check_for_win(self):
        if(any("D" in sublist for sublist in self.board)):
            self.winner = Winner.white#black
            self.done = True
            return
        if(any("Z" in sublist2 for sublist2 in self.board)): # Finds
            self.winner = Winner.black#white
            self.done = True
            return

    def _resigned(self):
        if self.player_turn() == Player.black: #white:
            self.winner = Winner.white
        else:
            self.winner = Winner.black
        self.done = True
        self.resigned = True

    def black_and_white_plane(self):
        board_white = np.copy(self.board)
        board_black = np.copy(self.board)
        board_blocks = np.copy(self.board)
        board_turn = np.copy(self.board)
        for i in range(5):
            for j in range(5):
                if(self.turn_n()%2==0):
                    board_turn[i][j] = 1
                else:
                    board_turn[i][j] = 0
        for i in range(5):
            for j in range(5):
                if self.board[i][j] == ' ':
                    board_white[i][j] = 0
                    board_black[i][j] = 0
                    board_blocks[i][j] = 0
                elif self.board[i][j] == 'A':
                    board_white[i][j] = 1
                    board_black[i][j] = 0
                    board_blocks[i][j] = 0
                elif self.board[i][j] == 'B':
                    board_white[i][j] = 1
                    board_black[i][j] = 0
                    board_blocks[i][j] = 1
                elif self.board[i][j] == 'C':
                    board_white[i][j] = 1
                    board_black[i][j] = 0
                    board_blocks[i][j] = 2
                elif self.board[i][j] == 'D':
                    board_white[i][j] = 1
                    board_black[i][j] = 0
                    board_blocks[i][j] = 3
                elif self.board[i][j] == 'W':
                    board_white[i][j] = 0
                    board_black[i][j] = 1
                    board_blocks[i][j] = 0
                elif self.board[i][j] == 'X':
                    board_white[i][j] = 0
                    board_black[i][j] = 1
                    board_blocks[i][j] = 1
                elif self.board[i][j] == 'Y':
                    board_white[i][j] = 0
                    board_black[i][j] = 1
                    board_blocks[i][j] = 2
                elif self.board[i][j] == 'Z':
                    board_white[i][j] = 0
                    board_black[i][j] = 1
                    board_blocks[i][j] = 3
                elif self.board[i][j] == '1':
                    board_white[i][j] = 0
                    board_black[i][j] = 0
                    board_blocks[i][j] = 1
                elif self.board[i][j] == '2':
                    board_white[i][j] = 0
                    board_black[i][j] = 0
                    board_blocks[i][j] = 2
                elif self.board[i][j] == '3':
                    board_white[i][j] = 0
                    board_black[i][j] = 0
                    board_blocks[i][j] = 3
                elif self.board[i][j] == '4':
                    board_white[i][j] = 0
                    board_black[i][j] = 0
                    board_blocks[i][j] = 4
        return np.array(board_white), np.array(board_black), np.array(board_blocks), np.array(board_turn)


    def render(self):
        print("\nRound: " + str(self.turn))

        temp=''
        for i in range(5):
            for j in range(5):
                if(self.board[i][j]==' '): temp+=('.')
                else: temp+=self.board[i][j]
            temp+='\n'
        print(temp)

        if self.done:
            print("Game Over!")
            if self.winner == Winner.white:
                print("White is the winner")
            elif self.winner == Winner.black:
                print("Black is the winner")

    @property
    def observation(self):
        return ''.join(''.join(x for x in y) for y in self.board)
