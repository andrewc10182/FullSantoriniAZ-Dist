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
    return EvolverWorker(config).start()

class EvolverWorker:
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
        self.play_files_per_generation = 15 # each file this number of games
        self.nb_plays_per_file = 10
        self.generations_to_keep = 20
        #self.min_play_files_to_learn = 0
        self.play_files_on_dropbox = 0
    def start(self):
        auth_token = 'UlBTypwXWYAAAAAAAAAAEP6hKysZi9cQKGZTmMu128TYEEig00w3b3mJ--b_6phN'
        self.dbx = dropbox.Dropbox(auth_token)  
        self.version = len(self.dbx.files_list_folder('/model/HistoryVersion').entries)
        print('\nThe Strongest Version found is: ',self.version,'\n')
        
        # Load either the latest ng model or the best model as self model
        self.model = self.load_model()
        self.compile_model()
            
        while True:
            if(self.dbx.files_list_folder('/state').entries[0].name == 'selfplaying'):
                
                self.play_files_on_dropbox = len(self.dbx.files_list_folder('/play_data').entries)

                target = min(int(self.dbx.files_list_folder('/target').entries[0].name),
                             self.generations_to_keep * self.play_files_per_generation)
                print('\nSelf-Play Files',self.play_files_on_dropbox,'out of',target,'\n')

                #while self.play_files_on_dropbox < self.min_play_files_to_learn:
                #    print('\nPlay Files Found:',self.play_files_on_dropbox,'of required',self.min_play_files_to_learn,'files. Started Self-Playing...\n')
                while self.play_files_on_dropbox < target:
                    self.self_play()
                    self.play_files_on_dropbox = len(self.dbx.files_list_folder('/play_data').entries)
                    print('\nSelf-Play Files',self.play_files_on_dropbox,'out of',target,'\n')
                #    self.play_files_on_dropbox = len(self.dbx.files_list_folder('/play_data').entries)
                #print('\nPlay Files Found:',self.play_files_on_dropbox,'of required',self.min_play_files_to_learn,'files. Training files sufficient for Learning!\n')
                  
                self.dbx.files_delete('/state/selfplaying')
                res = self.dbx.files_upload(bytes('abc', 'utf8'), '/state/training', dropbox.files.WriteMode.add, mute=True)   
            
            elif(self.dbx.files_list_folder('/state').entries[0].name == 'training'):
                # Training
                self.load_play_data()
                
                self.training()
            
                # Remove all Win Lose Records and start new again
                for entry in self.dbx.files_list_folder('/EvaluateWinCount').entries:
                    self.dbx.files_delete('/EvaluateWinCount/'+entry.name)
                    
                try: self.dbx.files_delete('/state/training')
                except: dummy=0
                res = self.dbx.files_upload(bytes('abc', 'utf8'), '/state/evaluating', dropbox.files.WriteMode.add, mute=True)

            elif(self.dbx.files_list_folder('/state').entries[0].name == 'evaluating'):
                # Evaluating                
                print('\nLoading Best Model:')
                self.best_model = self.load_best_model()
                RetrainSuccessful = self.evaluate()

                # Remove the oldest files if files is already Files per Gen x Generations to keep
                list = []
                for entry in self.dbx.files_list_folder('/play_data').entries:
                    list.append(entry)
                if(len(list)==self.play_files_per_generation * self.generations_to_keep):
                    for i in range(0,self.play_files_per_generation,1): #Remove the oldest 15 files i
