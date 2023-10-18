import pickle
from abc import ABC, abstractmethod
from threading import Lock, Thread
from queue import Queue, LifoQueue, Empty, Full
from time import time
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.planning.planners import MotionPlanner, NO_COUNTERS_PARAMS
from human_aware_rl.rllib.rllib import load_agent
import random, os, pickle, json
import ray


with open("/home/anacarrasco/overcooked-demo/server/static/assets/agents/RandAI/agent.pickle", "rb") as f:
    agent = pickle.load(f)

print(str(agent))