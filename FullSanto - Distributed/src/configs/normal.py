class EvaluateConfig:
    def __init__(self):
        self.game_num = 100
        self.replace_rate = 0.6
        self.play_config = PlayConfig()
        self.play_config.simulation_num_per_move = 4 #800
        self.play_config.thinking_loop = 1 #2
        self.play_config.c_puct = 1.5
        self.play_config.change_tau_turn = 0
        self.play_config.noise_eps = 0
        self.evaluate_latest_first = True
        self.next_generation_replace_rate = 0.55
        
class PlayDataConfig:
    def __init__(self):
        #self.nb_game_in_file = 10
        self.max_file_num = 5000

class PlayConfig:
    def __init__(self):
        self.simulation_num_per_move = 4 #800
        self.thinking_loop = 1
        self.logging_thinking = False
        self.c_puct = 1.5 #2
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.03
        self.change_tau_turn = 6
        self.virtual_loss = 3
        self.prediction_queue_size = 16
        self.parallel_search_num = 8 #2
        self.prediction_worker_sleep_sec = 0.0001
        self.wait_for_expanding_sleep_sec = 0.00001

class TrainerConfig:
    def __init__(self):
        self.batch_size = 1024  # 1024
        self.epoch_to_checkpoint = 1 #2
        self.start_total_steps = 0
        self.save_model_steps = 9999 # 150
        self.load_data_steps = 9999 # 150

class ModelConfig:
    def __init__(self):
        self.cnn_filter_num = 128
        self.cnn_filter_size = 3
        self.res_layer_num = 8
        self.l2_reg = 1e-4
        self.value_fc_size = 256
