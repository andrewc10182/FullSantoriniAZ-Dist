import os, time
import random
from datetime import datetime
import dropbox

import keras.backend as K
import numpy as np
from keras.optimizers import SGD

from agent.model import GameModel, objective_function_for_policy, objective_function_for_value
from agent.player import GamePlayer

from config import Config
from src.lib import tf_util
from src.lib.data_helper import get_game_data_filenames, write_game_data_to_file, read_game_data_from_file, get_next_generation_model_dirs
from src.lib.model_helpler import save_as_best_model, load_best_model_weight
from env.game_env import GameEnv, Player, Winner

def start(config: Config):
    tf_util.set_session_config(per_process_gpu_memory_fraction=0.59)
    return AssistantWorker(config).start()

class AssistantWorker:
    def __init__(self, config: Config):
        self.config = config
        self.model = None  # type: GameModel
        self.loaded_filenames = set()
        self.loaded_data = {}
        self.dataset = None
        self.optimizer = None
        self.dbx = None
        self.version = 0 # Change to dynamic lookup from Drop Box Files
        self.env = GameEnv()
        self.best_is_white = True
        self.play_files_per_generation = 28 # each file this number of games
        self.nb_plays_per_file = 25
        self.generations_to_keep = 10
        #self.min_play_files_to_learn = 0
        self.play_files_on_dropbox = 0
    def start(self):
        auth_token = 'UlBTypwXWYAAAAAAAAAAEP6hKysZi9cQKGZTmMu128TYEEig00w3b3mJ--b_6phN'
        self.dbx = dropbox.Dropbox(auth_token)  
        self.version = len(self.dbx.files_list_folder('/model/HistoryVersion').entries)
        print('\nThe Strongest Version found is: ',self.version,'\n')

        while True:
            if(self.dbx.files_list_folder('/state').entries[0].name == 'selfplaying'):
                # Try to empty the next generation folder before downloading
                try: self.remove_model(get_next_generation_model_dirs(self.config.resource)[0])
                except: dummy = 0
                
                self.model = self.load_model()
                self.compile_model()
                
                target = min(int(self.dbx.files_list_folder('/target').entries[0].name),
                             self.generations_to_keep * self.play_files_per_generation)
                self.play_files_on_dropbox = len(self.dbx.files_list_folder('/play_data').entries)

                print('\nSelf-Play Files',self.play_files_on_dropbox,'out of',target,'\n')

                self.self_play()

            elif(self.dbx.files_list_folder('/state').entries[0].name == 'training'):
                try: self.remove_model(get_next_generation_model_dirs(self.config.resource)[0])
                except: pass
                time.sleep(60)  
            elif(self.dbx.files_list_folder('/state').entries[0].name == 'evaluating'):
                # Evaluating                
                # Load either the latest ng model or the best model as self model
                try: self.remove_model(get_next_generation_model_dirs(self.config.resource)[0])
                except: pass
            
                self.model = self.load_model()
                self.compile_model()
                
                print('\nLoading Best Model:')
                self.best_model = self.load_best_model()
                RetrainSuccessful = self.evaluate()
                time.sleep(60)

                self.dataset = None
                
    def self_play(self):
        self.buffer = []
        idx = 1

        for _ in range(self.nb_plays_per_file):
            #self.play_files_on_dropbox = len(self.dbx.files_list_folder('/play_data').entries)
            #self.min_play_files_to_learn = min(self.version + 1, self.generations_to_keep) * self.play_files_per_generation 
            #if(self.play_files_on_dropbox >= self.min_play_files_to_learn):
            #    print('Training files sufficient for Learning, ending Self-Play...')
            #    break
            start_time = time.time()            
            env = self.self_play_game(idx)
            end_time = time.time()
            print("Game",idx," Time=",(end_time - start_time)," sec, Turn=", env.turn, env.observation, env.winner)
            idx += 1
            
            try:
                self.play_files_on_dropbox = len(self.dbx.files_list_folder('/play_data').entries)
                target = min(int(self.dbx.files_list_folder('/target').entries[0].name),
                         self.generations_to_keep * self.play_files_per_generation)
                if(self.play_files_on_dropbox >= target):
                    print('\nSufficient Play Data available, main Evolver is training...')
                    time.sleep(60)
                    break
            except: 'Cannot connect to Dropbox, this try saved the day!'

    def load_model(self):            
         # If there's an existing next generation model, use it
        try:
            next_gen_filename = self.dbx.files_list_folder('/model/next_generation').entries[0].name
            os.makedirs('FullSantoriniAZ-Dist/FullSanto - Distributed/data/model/next_generation/'+next_gen_filename)
            config_filename = self.dbx.files_list_folder('/model/next_generation/'+next_gen_filename).entries[0].name
            weight_filename = self.dbx.files_list_folder('/model/next_generation/'+next_gen_filename).entries[1].name
            md, res = self.dbx.files_download('/model/next_generation/'+next_gen_filename+'/'+config_filename)
            with open('FullSantoriniAZ-Dist/FullSanto - Distributed/data/model/next_generation/'+next_gen_filename+'/'+config_filename, 'wb') as f:  
                f.write(res.content)
            md, res = self.dbx.files_download('/model/next_generation/'+next_gen_filename+'/'+weight_filename)
            with open('FullSantoriniAZ-Dist/FullSanto - Distributed/data/model/next_generation/'+next_gen_filename+'/'+weight_filename, 'wb') as f:  
                f.write(res.content)
        except: dummy=0
              
        # Copies Dropbox's Best Model & Best Config to docker fodler
        for entry in self.dbx.files_list_folder('/model').entries:
            if(entry.name!='HistoryVersion' and entry.name!='next_generation'):
                md, res = self.dbx.files_download('/model/'+entry.name)
                with open('FullSantoriniAZ-Dist/FullSanto - Distributed/data/model/'+entry.name, 'wb') as f:  
                #with open('./data/model/'+entry.name, 'wb') as f:  
                    f.write(res.content)

        from agent.model import GameModel
        model = GameModel(self.config)
        if self.config.opts.new or not load_best_model_weight(model):
            model.build()
            save_as_best_model(model)

        dirs = get_next_generation_model_dirs(self.config.resource)
        print('Dirs is',dirs)
        if not dirs:
            print("\nLoading Self.Model = Best Model...")
            if not load_best_model_weight(model):
                print("Best model can not loaded!")
        else:
            latest_dir = dirs[-1]
            print("\nLoading Self.Model = Next Generation Model...")
            config_path = os.path.join(latest_dir, self.config.resource.next_generation_model_config_filename)
            weight_path = os.path.join(latest_dir, self.config.resource.next_generation_model_weight_filename)
            model.load(config_path, weight_path)
        return model

    def compile_model(self):
        self.optimizer = SGD(lr=1e-2, momentum=0.9)
        losses = [objective_function_for_policy, objective_function_for_value]
        self.model.model.compile(optimizer=self.optimizer, loss=losses)

    def training(self):
        last_load_data_step = last_save_step = total_steps = self.config.trainer.start_total_steps
        
        try: self.remove_model(get_next_generation_model_dirs(self.config.resource)[0])
        except: dummy=0
            
        steps = self.train_epoch(self.config.trainer.epoch_to_checkpoint)
        total_steps += steps
        self.save_current_model()
        last_save_step = total_steps

    def load_play_data(self):
        for entry in self.dbx.files_list_folder('/play_data').entries:
            md, res = self.dbx.files_download('/play_data/'+entry.name)
            with open('FullSantoriniAZ-Dist/FullSanto - Distributed/data/play_data/'+entry.name, 'wb') as f:  
            #with open('./data/play_data/'+entry.name, 'wb') as f:  
                f.write(res.content)
        filenames = get_game_data_filenames(self.config.resource)
        
        updated = False
        for filename in filenames:
            if filename not in self.loaded_filenames:
                self.load_data_from_file(filename)
            updated = True

        for filename in (self.loaded_filenames - set(filenames)):
            self.unload_data_of_file(filename)

        if updated:
            print("Updated Play Data.\n")
            self.dataset = self.collect_all_loaded_data()

    def load_data_from_file(self, filename):
        try:
            print("loading data from ",filename)
            data = read_game_data_from_file(filename)
            self.loaded_data[filename] = self.convert_to_training_data(data)
            self.loaded_filenames.add(filename)
        except Exception as e:
            print(str(e))

    def unload_data_of_file(self, filename):
        print("removing data about ",filename," from training set")
        self.loaded_filenames.remove(filename)
        if filename in self.loaded_data:
            del self.loaded_data[filename]

    def collect_all_loaded_data(self):
        state_ary_list, policy_ary_list, z_ary_list = [], [], []
        for s_ary, p_ary, z_ary_ in self.loaded_data.values():
            state_ary_list.append(s_ary)
            policy_ary_list.append(p_ary)
            z_ary_list.append(z_ary_)

        state_ary = np.concatenate(state_ary_list)
        policy_ary = np.concatenate(policy_ary_list)
        z_ary = np.concatenate(z_ary_list)
        return state_ary, policy_ary, z_ary

    @staticmethod
    def convert_to_training_data(data):
        state_list = []
        policy_list = []
        z_list = []
        for state, policy, z in data:
            board = list(state)
            board = np.reshape(board, (5, 5))
            env = GameEnv().update(board)

            white_ary, black_ary, block_ary, turn_ary = env.black_and_white_plane()
            state = [white_ary, black_ary, block_ary, turn_ary]

            state_list.append(state)
            policy_list.append(policy)
            z_list.append(z)

        return np.array(state_list), np.array(policy_list), np.array(z_list)

    @property
    def dataset_size(self):
        if self.dataset is None:
            return 0
        return len(self.dataset[0])

    def train_epoch(self, epochs):
        tc = self.config.trainer
        state_ary, policy_ary, z_ary = self.dataset
        self.model.model.fit(state_ary, [policy_ary, z_ary],
                             batch_size=tc.batch_size,
                             epochs=epochs, verbose=1, shuffle=True)
        steps = (state_ary.shape[0] // tc.batch_size) * epochs
        return steps

    def save_current_model(self):
        rc = self.config.resource
        model_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        model_dir = os.path.join(rc.next_generation_model_dir, rc.next_generation_model_dirname_tmpl % model_id)
        os.makedirs(model_dir, exist_ok=True)
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        self.model.save(config_path, weight_path)

    def load_best_model(self):
        model = GameModel(self.config)
        load_best_model_weight(model)
        return model

    def evaluate(self):
        ng_model, model_dir = self.load_next_generation_model()
        print("start evaluate model", model_dir)
        ng_is_great = self.evaluate_model(ng_model)
        
        return ng_is_great

    def load_next_generation_model(self):
        rc = self.config.resource
        while True:
            dirs = get_next_generation_model_dirs(self.config.resource)
            if dirs:
                break
        model_dir = dirs[-1] if self.config.eval.evaluate_latest_first else dirs[0]
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        model = GameModel(self.config)
        model.load(config_path, weight_path)
        return model, model_dir

    def evaluate_model(self, ng_model):
        ##results = []
        winning_rate = 0
        #for game_idx in range(1,self.config.eval.game_num+1):
        while(len(self.dbx.files_list_folder('/EvaluateWinCount').entries) < self.config.eval.game_num):
            
            # see if this will break out of evaluating
            if(self.dbx.files_list_folder('/state').entries[0].name == 'training'):
                break
            elif(self.dbx.files_list_folder('/state').entries[0].name == 'selftraining'):
                break
                
            ng_win, white_is_best = self.play_game(self.best_model, ng_model)
            
            # Save a "Win" File in Dropbox if win, and "Lose" File if lose
            if(ng_win==1): 
                if(white_is_best): filename = 'wb'+str(random.random()*2000000//2)
                else: filename = 'ww'+str(random.random()*2000000//2)
            else: 
                if(white_is_best): filename = 'lb'+str(random.random()*2000000//2)
                else: filename = 'lw'+str(random.random()*2000000//2)
                
            res = self.dbx.files_upload(bytes('abc', 'utf8'), '/EvaluateWinCount/'+filename, dropbox.files.WriteMode.add, mute=True)
            
            ww = 0
            wb = 0
            lw = 0
            lb = 0
            
            try:
                for entry in self.dbx.files_list_folder('//EvaluateWinCount').entries:
                    if(entry.name[:2] == 'ww'): ww +=1
                    elif(entry.name[:2] == 'wb'): wb += 1
                    elif(entry.name[:2] == 'lw'): lw += 1
                    elif(entry.name[:2] == 'lb'): lb += 1
                temp = 'In '+str(ww+wb+lw+lb)+' Cloud Games, NG Wins '+str(ww)+' of '+str(ww+lw)+' (%.3f)%% games in White, ' + str(wb)+' of '+str(wb+lb)+' (%.3f)%% games in Black, '+ str(ww+wb)+' of '+str(ww+wb+lw+lb)+' (%.3f)%% games in Total.'
                temp2 = (ww/(ww+lw)*100, wb/(wb+lb)*100, (ww+wb)/(ww+wb+lw+lb)*100)
                print(temp % temp2)
            except: pass
            
        return True

    def play_game(self, best_model, ng_model):
        env = GameEnv().reset()
    
        best_player = GamePlayer(self.config, best_model, play_config=self.config.eval.play_config)
        ng_player = GamePlayer(self.config, ng_model, play_config=self.config.eval.play_config)
        self.best_is_white = not self.best_is_white
        if not self.best_is_white:
            black, white = best_player, ng_player
        else:
            black, white = ng_player, best_player

        env.reset()
        while not env.done:
            if env.player_turn() == Player.black:
                action = black.action(env.board)
            else:
                action = white.action(env.board)
            env.step(action)

        ng_win = None
        if env.winner == Winner.white:
            if self.best_is_white:
                ng_win = 0
            else:
                ng_win = 1
        elif env.winner == Winner.black:
            if self.best_is_white:
                ng_win = 1
            else:
                ng_win = 0
        return ng_win, self.best_is_white

    def remove_model(self, model_dir):
        rc = self.config.resource
        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        os.remove(config_path)
        os.remove(weight_path)
        os.rmdir(model_dir)

    def self_play_game(self, idx):
        self.env.reset()
        self.black = GamePlayer(self.config, self.model)
        self.white = GamePlayer(self.config, self.model)
        while not self.env.done:
            if self.env.player_turn() == Player.black:
                action = self.black.action(self.env.board)
            else:
                action = self.white.action(self.env.board)
            self.env.step(action)
            
            # Do again if White won
            #if self.env.winner == Winner.black:
            #    continue
            #elif self.env.winner == Winner.white:
            #    self.env.reset()
            #    self.black.moves = []
            #    self.white.moves = []
                
        self.finish_game()
        self.save_play_data(write=idx % self.nb_plays_per_file == 0)
        self.remove_play_data()
        return self.env

    def finish_game(self):
        if self.env.winner == Winner.black:
            black_win = 1
        elif self.env.winner == Winner.white:
            black_win = -1
        else:
            black_win = 0
        self.black.finish_game(black_win)
        self.white.finish_game(-black_win)

    def save_play_data(self, write=True):
        data = self.black.moves + self.white.moves
        self.buffer += data

        if not write:
            return

        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        filename = rc.play_data_filename_tmpl % game_id
        path = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % game_id)
        print("save play data to ",path)
        write_game_data_to_file(path, self.buffer)
        
        # Saving File to Drop Box
        target = min(int(self.dbx.files_list_folder('/target').entries[0].name),
             self.generations_to_keep * self.play_files_per_generation)
        self.play_files_on_dropbox = len(self.dbx.files_list_folder('/play_data').entries)
        
        if(self.play_files_on_dropbox < target):
            print('Contributing self-play games to Dropbox...')
            with open(path, 'rb') as f:
                data = f.read()
            res = self.dbx.files_upload(data, '/play_data/'+filename, dropbox.files.WriteMode.add, mute=True)
        else:
            print('Sufficient games are in Dropbox.  Self Play games not saved.  Waiting for Training...')
            time.sleep(60)
        self.buffer = []

    def remove_play_data(self):
        files = get_game_data_filenames(self.config.resource)
        if len(files) < self.play_files_per_generation*self.generations_to_keep:
            return
        for i in range(len(files) - self.play_files_per_generation*self.generations_to_keep):
            os.remove(files[i])
            
        # Also remove the oldest 15 files from dropbox
        #    localfilenames = []
        #        for a in range(len(files)):
        #            localfilenames.append(files[a][-32:])
        #        dbfiles = []
        #        for entry in self.dbx.files_list_folder('/play_data').entries:
        #            dbfiles.append(entry.name)
        #        localfiles_to_remove = set(localfilenames) - set(dbfiles)
        #        print('Removing',len(localfiles_to_remove),'files from local drive')
        #        for file in localfiles_to_remove:
        #            print('Removing local play_data file',file)
        #            path = os.path.join(self.config.resource.play_data_dir,file)
        #            os.remove(path)
        #        break
            
    def remove_all_play_data(self):
        files = get_game_data_filenames(self.config.resource)
        for i in range(len(files)):
            os.remove(files[i])
