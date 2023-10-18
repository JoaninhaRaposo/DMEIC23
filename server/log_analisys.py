# import pygame module in this program

#from teammates.Astro import ACTION_MEANINGS_MDP 
import itertools
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
PATH = os.path.abspath(os.getcwd())

ACTION_MEANING = [
    "move to lower-index node",
    "move to second-lower-index node",
    "move to third-lower-index node",
    "move to fourth-lower-index node",
    "stay",
    "act"]

log = []
LVL=1
#LOG_NRS = range(300,329, 2) # PICK CONDITION: nrs impares - condição 1 | nrs pares - condição 2
#LOG_NRS = range(301,330, 2) #cond 1
#LOG_NRS = range(400, 402, 1)
#ACTION_SPACE = tuple(range(len(ACTION_MEANING[LVL-1])))
ACTION_SPACE = tuple(range(len(ACTION_MEANING)))
JOINT_ACTION_SPACE = list(itertools.product(ACTION_SPACE, repeat=2))
#print("READING FROM: ", log_file)

class LogFrame:
  def __init__(self, timestep:int , state_env, time_ball:float, game_time:float):#, state_mdp:list, action_env:tuple, action_mdp:tuple:
    self.timestep = timestep
    self.state_env = state_env
    #self.state_mdp = state_mdp
    #self.action_env = action_env
    #self.action_mdp = action_mdp
    #self.onion_time = onion_time
    self.game_time = game_time
    self.time_ball = time_ball

def count_occurrences(arrays):
    return Counter([tuple(array) for array in arrays])

s_env = []
s_mdp = []
s_full_env = []
ball_time = []
game_time = []
total_scores = []
location = "logfiles/lvl1/real_ai_agent/cond_Low/"
#for i in LOG_NRS:
for file in os.listdir(location):
    #print("{}".format(file))
    # Concatenate all logfiles from chosen condition
    log_file = f"logfiles/lvl1/real_ai_agent/cond_Low/{file}"
    with open(log_file, "rb") as f:
        log = pickle.load(f)

    for logframe in log:
        #s_env.append(logframe.state_env[:4])
        #s_mdp.append(logframe.state_mdp)
        s_full_env.append(logframe.state_env)
    
    ball_time.append(logframe.time_ball)
    game_time.append(logframe.game_time)


print("State objects")
for state in s_full_env:
    print(state['players'][0]['held_object'])

"""with open(f"{PATH}/log_results.txt", "w") as f: #mudar para modo append
            f.write(str("Ball time"))
            f.close()
"""
print("Ball Time")

for time in ball_time:
    
    """with open(f"{PATH}/log_results.txt", "w") as f:
            f.write(str(time))
            f.close()"""
    print(time)

with open(f"{PATH}/log_results.txt", "w") as f:
            f.write("Game time")
            f.close()

print("Game Time")
for time in game_time:
    """with open(f"{PATH}/log_results.txt", "w") as f:
            f.write(str(time))
            f.close()"""
    print(time)
"""with open(f"{PATH}/log_results.txt", "w") as f:
            f.write("Score")
            f.close()"""
print("Score")
for i in range(len(game_time)):
    score = 100 - game_time[i] - round(ball_time[i])
    """with open(f"{PATH}/log_results.txt", "w") as f:
            f.write(str(score))
            f.close()"""
    print(score)
