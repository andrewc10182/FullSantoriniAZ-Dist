from config import Config, PlayWithHumanConfig
from play_game.game_model import PlayWithHuman
from env.game_env import GameEnv, Player, Winner
from random import random

def start(config: Config):
    PlayWithHumanConfig().update_play_config(config.play)
    game_model = PlayWithHuman(config)

    while True:
        env = GameEnv().reset()
        human_is_black = random() < 0.5
        game_model.start_game(human_is_black)

        while not env.done:
            if env.player_turn() == Player.black:
                if not human_is_black:
                    action = game_model.move_by_ai(env)
                    print("IA moves to: " + str(action))
                else:
                    action = game_model.move_by_human(env)
                    print("You move to: " + str(action))
            else:
                if human_is_black:
                    action = game_model.move_by_ai(env)
                    print("IA moves to: " + str(action))
                else:
                    action = game_model.move_by_human(env)
                    print("You move to: " + str(action))
            env.step(action)
            env.render()

        print("\nEnd of the game.")
        print("Game result:")
        if env.winner == Winner.white:
            print("X wins")
        elif env.winner == Winner.black:
            print("O wins")
        else:
            print("Game was a draw")
