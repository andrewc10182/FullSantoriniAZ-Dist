from asyncio import Future
from asyncio.queues import Queue
from collections import defaultdict, namedtuple
import asyncio

import numpy as np

from agent.model_api import ModelAPI
from config import Config
from env.game_env import GameEnv, Winner, Player

CounterKey = namedtuple("CounterKey", "board next_player")
QueueItem = namedtuple("QueueItem", "state future")
HistoryItem = namedtuple("HistoryItem", "action policy values visit")

class GamePlayer:
    def __init__(self, config: Config, model, play_config=None):

        self.config = config
        self.model = model
        self.play_config = play_config or self.config.play
        self.api = ModelAPI(self.config, self.model)

        self.labels_n = config.n_labels
        self.var_n = defaultdict(lambda: np.zeros((self.labels_n,)))
        self.var_w = defaultdict(lambda: np.zeros((self.labels_n,)))
        self.var_q = defaultdict(lambda: np.zeros((self.labels_n,)))
        self.var_u = defaultdict(lambda: np.zeros((self.labels_n,)))
        self.var_p = defaultdict(lambda: np.zeros((self.labels_n,)))
        self.expanded = set()
        self.now_expanding = set()
        self.prediction_queue = Queue(self.play_config.prediction_queue_size)
        self.sem = asyncio.Semaphore(self.play_config.parallel_search_num)

        self.moves = []
        self.loop = asyncio.get_event_loop()
        self.running_simulation_num = 0

        self.thinking_history = {}  # for fun

    def action(self, board):
        env = GameEnv().update(board)
        key = self.counter_key(env)

        No_Black = True
        Black = ['W', 'X', 'Y']
        for i in range(5):
            for j in range(5):
                if env.board[i][j] in Black:
                    No_Black = False
        if No_Black: action = 999
        else:
            for tl in range(self.play_config.thinking_loop):
                self.search_moves(board)

                policy = self.calc_policy(board)
                action = int(np.random.choice(range(self.labels_n), p=policy))

                action_by_value = int(np.argmax(self.var_q[key] + (self.var_n[key] > 0)*100))
                if action == action_by_value or env.turn < self.play_config.change_tau_turn:
                    break

            # this is for play_gui, not necessary when training.
            self.thinking_history[env.observation] = HistoryItem(action, policy, list(self.var_q[key]), list(self.var_n[key]))
            mirror_x_env, mirror_x_policy = self.mirror_x(env, list(policy))
            mirror_y_env, mirror_y_policy = self.mirror_y(env, list(policy))
            mirror_xy_env, mirror_xy_policy = self.mirror_y(mirror_x_env, mirror_x_policy)
            mirror_d_env, mirror_d_policy = self.mirror_d(env, list(policy))
            mirror_dx_env, mirror_dx_policy = self.mirror_x(mirror_d_env, mirror_d_policy)
            mirror_dy_env, mirror_dy_policy = self.mirror_y(mirror_d_env, mirror_d_policy)
            mirror_dxy_env, mirror_dxy_policy = self.mirror_y(mirror_dx_env, mirror_dx_policy)
            self.moves.append([env.observation, list(policy)])
            self.moves.append([mirror_x_env.observation, mirror_x_policy])
            self.moves.append([mirror_y_env.observation, mirror_y_policy])
            self.moves.append([mirror_xy_env.observation, mirror_xy_policy])
            self.moves.append([mirror_d_env.observation, mirror_d_policy])
            self.moves.append([mirror_dx_env.observation, mirror_dx_policy])
            self.moves.append([mirror_dy_env.observation, mirror_dy_policy])
            self.moves.append([mirror_dxy_env.observation, mirror_dxy_policy])
        return action
    
    def mirror_x(self, env, policy):
        newenv = GameEnv().reset()
        for i in range(5):
            for j in range(5):
                newenv.board[i][j] = env.board[4-i][j]
        player = ['A', 'B', 'C']
        if(env.player_turn() == Player.black): player = ['W', 'X', 'Y']
        
        newpolicy = [0]*128
        for action in range(len(policy)):
            if(policy[action]==0):
                continue
            else:
                moving_worker = action // 64 # 0 if first worker, 1 if second worker
                board = list(np.reshape(env.board,25))
                worker1_pos = [i for i, n in enumerate(board) if n in player][0]
                worker2_pos = [i for i, n in enumerate(board) if n in player][1]

                if(moving_worker == 0):
                    if(worker1_pos//5 < worker2_pos//5):
                        moving_worker = 1-moving_worker
                else:
                    if(worker2_pos//5 > worker1_pos//5):
                        moving_worker = 1-moving_worker
                      
                MoveCode = int((action%64)/8)+1
                if(MoveCode>4): MoveCode+=1 #Changes 5 to 8 into 6 to 9

                if(MoveCode == 1): MoveCode = 7 # text = text + 'Move ↖ '
                elif(MoveCode == 2): MoveCode = 8 #text = text + 'Move ↑ '
                elif(MoveCode == 3): MoveCode = 9 #text = text + 'Move ↗ '
                elif(MoveCode == 7): MoveCode = 1
                elif(MoveCode == 8): MoveCode = 2
                elif(MoveCode == 9): MoveCode = 3

                BuildCode = int(action%8)+1
                if(BuildCode>4): BuildCode+=1
                
                if(BuildCode == 1): BuildCode = 7 #text = text + 'Build ↖'
                elif(BuildCode == 2): BuildCode = 8 #text = text + 'Build ↑'
                elif(BuildCode == 3): BuildCode = 9 #text = text + 'Build ↗'
                elif(BuildCode == 7): BuildCode = 1 #text = text + 'Build ↙'
                elif(BuildCode == 8): BuildCode = 2 #text = text + 'Build ↓'
                elif(BuildCode == 9): BuildCode = 3 #text = text + 'Build ↘'

                if(MoveCode>4): MoveCode-=1 #Changes 6 to 9 into 5 to 8
                if(BuildCode>4): BuildCode-=1
                newaction = (MoveCode-1)*8 + BuildCode - 1 + moving_worker*64
                #print('Action converted from',action,'to',newaction)
                newpolicy[newaction] = policy[action]

        return newenv, newpolicy

    def mirror_y(self, env, policy):
        newenv = GameEnv().reset()
        for i in range(5):
            for j in range(5):
                newenv.board[i][j] = env.board[i][4-j]
        player = ['A', 'B', 'C']
        if(env.player_turn() == Player.black): player = ['W', 'X', 'Y']
        
        newpolicy = [0]*128
        for action in range(len(policy)):
            if(policy[action]==0):
                continue
            else:
                moving_worker = action // 64 # 0 if first worker, 1 if second worker
                board = list(np.reshape(env.board,25))
                worker1_pos = [i for i, n in enumerate(board) if n in player][0]
                worker2_pos = [i for i, n in enumerate(board) if n in player][1]

                if(worker1_pos//5==worker2_pos//5):
                    moving_worker = 1-moving_worker
                      
                MoveCode = int((action%64)/8)+1
                if(MoveCode>4): MoveCode+=1 #Changes 5 to 8 into 6 to 9

                if(MoveCode == 1): MoveCode = 3 
                elif(MoveCode == 4): MoveCode = 6 
                elif(MoveCode == 7): MoveCode = 9 
                elif(MoveCode == 3): MoveCode = 1
                elif(MoveCode == 6): MoveCode = 4
                elif(MoveCode == 9): MoveCode = 7

                BuildCode = int(action%8)+1
                if(BuildCode>4): BuildCode+=1
                
                if(BuildCode == 1): BuildCode = 3
                elif(BuildCode == 4): BuildCode = 6 
                elif(BuildCode == 7): BuildCode = 9 
                elif(BuildCode == 3): BuildCode = 1 
                elif(BuildCode == 6): BuildCode = 4 
                elif(BuildCode == 9): BuildCode = 7 

                if(MoveCode>4): MoveCode-=1 #Changes 6 to 9 into 5 to 8
                if(BuildCode>4): BuildCode-=1
                newaction = (MoveCode-1)*8 + BuildCode - 1 + moving_worker*64
                #print('Action converted from',action,'to',newaction)
                newpolicy[newaction] = policy[action]

        return newenv, newpolicy

    def d_new_pos(self, pos):
        if pos==0: return 0
        elif pos==1: return 5
        elif pos==2: return 10
        elif pos==3: return 15
        elif pos==4: return 20
        elif pos==5: return 1
        elif pos==6: return 6
        elif pos==7: return 11
        elif pos==8: return 16
        elif pos==9: return 21
        elif pos==10: return 2
        elif pos==11: return 7
        elif pos==12: return 12
        elif pos==13: return 17
        elif pos==14: return 22
        elif pos==15: return 3
        elif pos==16: return 8
        elif pos==17: return 13
        elif pos==18: return 18
        elif pos==19: return 23
        elif pos==20: return 4
        elif pos==21: return 9
        elif pos==22: return 14
        elif pos==23: return 19
        elif pos==24: return 24

    def first_worker_is_first(self, worker1_pos, worker2_pos):
        if(worker1_pos//5 < worker2_pos//5):
            return True
        elif(worker1_pos < worker2_pos):
            return True
        return False

    def mirror_d(self, env, policy):
        newenv = GameEnv().reset()
        for i in range(5):
            for j in range(5):
                newenv.board[i][j] = env.board[j][i]
        player = ['A', 'B', 'C']
        if(env.player_turn() == Player.black): player = ['W', 'X', 'Y']
        
        newpolicy = [0]*128
        for action in range(len(policy)):
            if(policy[action]==0):
                continue
            else:
                moving_worker = action // 64 # 0 if first worker, 1 if second worker
                board = list(np.reshape(env.board,25))
                worker1_pos = [i for i, n in enumerate(board) if n in player][0]
                worker2_pos = [i for i, n in enumerate(board) if n in player][1]

                if(self.first_worker_is_first(self.d_new_pos(worker1_pos), self.d_new_pos(worker2_pos)) == False):
                    moving_worker = 1-moving_worker
                      
                MoveCode = int((action%64)/8)+1
                if(MoveCode>4): MoveCode+=1 #Changes 5 to 8 into 6 to 9

                if(MoveCode == 2): MoveCode = 4 # text = text + 'Move ↖ '
                elif(MoveCode == 3): MoveCode = 7 #text = text + 'Move ↑ '
                elif(MoveCode == 6): MoveCode = 8 #text = text + 'Move ↗ '
                elif(MoveCode == 4): MoveCode = 2
                elif(MoveCode == 7): MoveCode = 3
                elif(MoveCode == 8): MoveCode = 6

                BuildCode = int(action%8)+1
                if(BuildCode>4): BuildCode+=1
                
                if(BuildCode == 2): BuildCode = 4 #text = text + 'Build ↖'
                elif(BuildCode == 3): BuildCode = 7 #text = text + 'Build ↑'
                elif(BuildCode == 6): BuildCode = 8 #text = text + 'Build ↗'
                elif(BuildCode == 4): BuildCode = 2 #text = text + 'Build ↙'
                elif(BuildCode == 7): BuildCode = 3 #text = text + 'Build ↓'
                elif(BuildCode == 8): BuildCode = 6 #text = text + 'Build ↘'

                if(MoveCode>4): MoveCode-=1 #Changes 6 to 9 into 5 to 8
                if(BuildCode>4): BuildCode-=1
                newaction = (MoveCode-1)*8 + BuildCode - 1 + moving_worker*64
                #print('Action converted from',action,'to',newaction)
                newpolicy[newaction] = policy[action]

        return newenv, newpolicy
    
    def policy_to_direct_text(self, action):
        moving_worker = action // 64
        text = 'Action '+str(action)+' '
        text = text + 'Worker '+str(moving_worker)+' '
        action = action % 64
        
        MoveCode = int(action/8)+1
        if(MoveCode>4): MoveCode+=1 #Changes 5 to 8 into 6 to 9
        if(MoveCode == 1):
            text = text + 'Move ↖ '
        if(MoveCode == 2):
            text = text + 'Move ↑ '
        if(MoveCode == 3):
            text = text + 'Move ↗ '
        if(MoveCode == 4):
            text = text + 'Move ← '
        if(MoveCode == 6):
            text = text + 'Move → '
        if(MoveCode == 7):
            text = text + 'Move ↙ '
        if(MoveCode == 8):
            text = text + 'Move ↓ '
        if(MoveCode == 9):
            text = text + 'Move ↘ '

        BuildCode = int(action%8)+1
        if(BuildCode>4): BuildCode+=1

        #print('BuildCode',BuildCode,'Pos i & j at ',Posi, Posj)
        if(BuildCode == 1): text = text + 'Build ↖'
        if(BuildCode == 2): text = text + 'Build ↑'
        if(BuildCode == 3): text = text + 'Build ↗'
        if(BuildCode == 4): text = text + 'Build ←'
        if(BuildCode == 6): text = text + 'Build →'
        if(BuildCode == 7): text = text + 'Build ↙'
        if(BuildCode == 8): text = text + 'Build ↓'
        if(BuildCode == 9): text = text + 'Build ↘'

        return text
    def ask_thought_about(self, board) -> HistoryItem:
        return self.thinking_history.get(board)

    def search_moves(self, board):
        loop = self.loop
        self.running_simulation_num = 0

        coroutine_list = []
        for it in range(self.play_config.simulation_num_per_move):
            cor = self.start_search_my_move(board)
            coroutine_list.append(cor)
        coroutine_list.append(self.prediction_worker())
        loop.run_until_complete(asyncio.gather(*coroutine_list))

    async def start_search_my_move(self, board):
        self.running_simulation_num += 1
        with await self.sem:
            env = GameEnv().update(board)
            leaf_v = await self.search_my_move(env, is_root_node=True)
            self.running_simulation_num -= 1
            return leaf_v

    async def search_my_move(self, env: GameEnv, is_root_node=False):
        """
        Q, V is value for this Player(always white).
        P is value for the player of next_player (black or white)
        :param env:
        :param is_root_node:
        :return:
        """
        if env.done:
            if env.winner == Winner.white:
                return 1 # White goes second
            elif env.winner == Winner.black:
                return -1 # Black goes first
            else:
                return 0

        key = self.counter_key(env)

        while key in self.now_expanding:
            await asyncio.sleep(self.config.play.wait_for_expanding_sleep_sec)

        # is leaf?
        if key not in self.expanded:  # reach leaf node
            leaf_v = await self.expand_and_evaluate(env)
            if env.player_turn() == Player.white:
                return leaf_v  # Value for white
            else:
                return -leaf_v  # Value for white == -Value for white

        action_t = self.select_action_q_and_u(env, is_root_node)
        _, _ = env.step(action_t)

        virtual_loss = self.config.play.virtual_loss
        self.var_n[key][action_t] += virtual_loss
        self.var_w[key][action_t] -= virtual_loss

        
        leaf_v = await self.search_my_move(env)  # next move

        # on returning search path update: N, W, Q, U
        n = self.var_n[key][action_t] = self.var_n[key][action_t] - virtual_loss + 1
        w = self.var_w[key][action_t] = self.var_w[key][action_t] + virtual_loss + leaf_v
        self.var_q[key][action_t] = w / n
        
        return leaf_v

    async def expand_and_evaluate(self, env):
        """expand new leaf
        update var_p, return leaf_v
        :param ChessEnv env:
        :return: leaf_v
        """
        key = self.counter_key(env)
        self.now_expanding.add(key)

        white_ary, black_ary , block_ary, turn_ary = env.black_and_white_plane()
        #state = [black_ary, white_ary, block_ary, turn_ary] if env.player_turn() == Player.black else [white_ary, black_ary, block_ary, turn_ary]
        state = [white_ary, black_ary, block_ary, turn_ary]
        
        future = await self.predict(np.array(state))  # type: Future
        await future
        leaf_p, leaf_v = future.result()

        self.var_p[key] = leaf_p  # P is value for next_player (black or white)
        self.expanded.add(key)
        self.now_expanding.remove(key)
        return float(leaf_v)

    async def prediction_worker(self):
        """For better performance, queueing prediction requests and predict together in this worker.
        speed up about 45sec -> 15sec for example.
        :return:
        """
        q = self.prediction_queue
        margin = 10  # avoid finishing before other searches starting.
        while self.running_simulation_num > 0 or margin > 0:
            if q.empty():
                if margin > 0:
                    margin -= 1
                await asyncio.sleep(self.config.play.prediction_worker_sleep_sec)
                continue
            item_list = [q.get_nowait() for _ in range(q.qsize())]  # type: list[QueueItem]
            data = np.array([x.state for x in item_list])

            policy_ary, value_ary = self.api.predict(data)
            
            for p, v, item in zip(policy_ary, value_ary, item_list):
                item.future.set_result((p, v))

    async def predict(self, x):
        future = self.loop.create_future()
        item = QueueItem(x, future)
        await self.prediction_queue.put(item)
        return future

    def finish_game(self, z):
        #param z: win=1, lose=-1, draw=0
        for move in self.moves:  # add this game winner result to all past moves.
            move += [z]

    def calc_policy(self, board):
        #calc π(a|s0)
        pc = self.play_config
        env = GameEnv().update(board)
        key = self.counter_key(env)
        if env.turn < pc.change_tau_turn:
            #print(key)
            #print('self.var_n[key]',self.var_n[key])
            #print('np.sum(self.var_n[key]',np.sum(self.var_n[key]))
            #input()
            return self.var_n[key] / np.sum(self.var_n[key])  # tau = 1
        else:
            action = np.argmax(self.var_n[key])  # tau = 0
            ret = np.zeros(self.labels_n)
            ret[action] = 1
            return ret

    @staticmethod
    def counter_key(env: GameEnv):
        return CounterKey(env.observation, env.turn)

    def select_action_q_and_u(self, env, is_root_node):
        key = self.counter_key(env)

        legal_moves = env.legal_moves()

        # noinspection PyUnresolvedReferences
        xx_ = np.sqrt(np.sum(self.var_n[key]))  # SQRT of sum(N(s, b); for all b)
        xx_ = max(xx_, 1)  # avoid u_=0 if N is all 0
        p_ = self.var_p[key]
        
        if is_root_node:
            p_ = (1 - self.play_config.noise_eps) * p_ + \
                 self.play_config.noise_eps * np.random.dirichlet([self.play_config.dirichlet_alpha] * self.labels_n)
        u_ = self.play_config.c_puct * p_ * xx_ / (1 + self.var_n[key])

        if env.player_turn() == Player.white:
            v_ = (self.var_q[key] + u_ + 1000) * legal_moves
        else:
            # When enemy's selecting action, flip Q-Value.
            v_ = (-self.var_q[key] + u_ + 1000) * legal_moves

        # noinspection PyTypeChecker
        action_t = int(np.argmax(v_))

        return action_t
