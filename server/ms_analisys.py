# import pygame module in this program

#from teammates.Astro import ACTION_MEANINGS_MDP 
import itertools
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
PATH = os.path.abspath(os.getcwd())

ACTION_MEANING = [
    "move to lower-index node",
    "move to second-lower-index node",
    "move to third-lower-index node",
    "move to fourth-lower-index node",
    "stay",
    "act"]

log = []
LVL=2



#COND = "High"
#COND = "Low"
COND = "Human"


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
total_hold_ball = []
hold_ball = 0
total_timesteps = []

states_holding_ball = []

location = f"{COND}_analysis/lvl{LVL}/"
write_log = f"timestep_analisys_cond_{COND}_lvl{LVL}.csv"

with open(write_log, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["ID", "Timesteps holding ball", "Total timesteps",	"Total different states holding ball"])

#for i in LOG_NRS:
for filename in os.listdir(location):
    #print("{}".format(file))

    # Concatenate all logfiles from chosen condition
    log_file = f"{location}/{filename}"
    with open(log_file, "rb") as f:
        log = pickle.load(f)

    hold_ball = 0
    for logframe in log:
        #s_env.append(logframe.state_env[:4])
        #s_mdp.append(logframe.state_mdp)
        s_full_env.append(logframe.state_env)
        if logframe.state_env['players'][0]['held_object'] is not None:
            hold_ball += 1

            state = logframe.state_env['players'][0]['position']
            if state not in states_holding_ball:
                 states_holding_ball.append(state)
        
    with open(write_log, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([filename,hold_ball,logframe.timestep,len(states_holding_ball)])

    

    total_hold_ball.append(hold_ball)
    ball_time.append(logframe.time_ball)
    game_time.append(logframe.game_time)
    total_timesteps.append(logframe.timestep)

print("timesteps holding ball")
sum = 0
for counter in total_hold_ball:
    sum = sum + counter    
    print(counter)

print("Holding ball timesteps average:")
print(sum/len(total_hold_ball))

sum = 0
print("Total timesteps")
for step in total_timesteps:
    sum = sum + step 
    print(step)
print("Total timesteps average")
print(sum/len(total_timesteps))


#print("Object States")
#for state in s_full_env:
#    print(state['objects'])

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
