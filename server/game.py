from abc import ABC, abstractmethod
import itertools
from threading import Lock, Thread
from queue import Queue, LifoQueue, Empty, Full
from time import time
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, ObjectState, OvercookedState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.planning.planners import MotionPlanner, NO_COUNTERS_PARAMS
from human_aware_rl.rllib.rllib import load_agent

from search2 import A_star_search_overcooked_wrapper
import random
import os
import copy
import pickle
import json
import ray
import numpy as np
import os
PATH = os.path.abspath(os.getcwd())
from overcooked_ai_py.utils import (
    read_layout_dict,
)

CURR_LAYOUT = "cramped"
CURR_CONDITION = None
ACTION_MEANING = [
    "move to lower-index node",
    "move to second-lower-index node",
    "move to third-lower-index node",
    "move to fourth-lower-index node",
    "stay",
    "act"]






# Relative path to where all static pre-trained agents are stored on server
AGENT_DIR = None

# Maximum allowable game time (in seconds)
MAX_GAME_TIME = None


SLIP = False


def _configure(max_game_time, agent_dir):
    global AGENT_DIR, MAX_GAME_TIME
    MAX_GAME_TIME = max_game_time
    AGENT_DIR = agent_dir


def fix_bc_path(path):
    """
    Loading a PPO agent trained with a BC agent requires loading the BC model as well when restoring the trainer, even though the BC model is not used in game
    For now the solution is to include the saved BC model and fix the relative path to the model in the config.pkl file
    """

    import pickle
    # the path is the agents/Rllib.*/agent directory
    agent_path = os.path.dirname(path)
    with open(os.path.join(agent_path, "config.pkl"), "rb") as f:
        data = pickle.load(f)
    bc_model_dir = data["bc_params"]["bc_config"]["model_dir"]
    last_dir = os.path.basename(bc_model_dir)
    bc_model_dir = os.path.join(agent_path, "bc_params", last_dir)
    data["bc_params"]["bc_config"]["model_dir"] = bc_model_dir
    with open(os.path.join(agent_path, "config.pkl"), "wb") as f:
        pickle.dump(data, f)



class LogFrame:
  def __init__(self, timestep:int , state_env, state_mdp:list, action_env:tuple, action_mdp:tuple, time_ball:float, game_time:float):
    self.timestep = timestep
    self.state_env = state_env
    self.state_mdp = state_mdp
    self.action_env = action_env
    self.action_mdp = action_mdp

class LogFrameTime:
    def __init__(self, time_ball: float, game_time: float):
        self.time_ball = time_ball
        self.game_time = game_time

class Game(ABC):

    """
    Class representing a game object. Coordinates the simultaneous actions of arbitrary
    number of players. Override this base class in order to use. 

    Players can post actions to a `pending_actions` queue, and driver code can call `tick` to apply these actions.


    It should be noted that most operations in this class are not on their own thread safe. Thus, client code should
    acquire `self.lock` before making any modifications to the instance. 

    One important exception to the above rule is `enqueue_actions` which is thread safe out of the box
    """

    # Possible TODO: create a static list of IDs used by the class so far to verify id uniqueness
    # This would need to be serialized, however, which might cause too great a performance hit to
    # be worth it

    EMPTY = 'EMPTY'


    class Status:
        DONE = 'done'
        ACTIVE = 'active'
        RESET = 'reset'
        INACTIVE = 'inactive'
        ERROR = 'error'

    def __init__(self, *args, **kwargs):
        """
        players (list): List of IDs of players currently in the game
        spectators (set): Collection of IDs of players that are not allowed to enqueue actions but are currently watching the game
        id (int):   Unique identifier for this game
        pending_actions List[(Queue)]: Buffer of (player_id, action) pairs have submitted that haven't been commited yet
        lock (Lock):    Used to serialize updates to the game state
        is_active(bool): Whether the game is currently being played or not
        """
        global CURR_CONDITION
        self.players = []
        self.spectators = set()
        self.pending_actions = []
        self.id = kwargs.get('id', id(self)) #pay attention here
    
        self.userid = kwargs.get('userid')#, id(self))  #Fixed bug
    
        #Select your condition
        #CURR_CONDITION = "High"
        #CURR_CONDITION = "Low"
        CURR_CONDITION = "Human" #Comment this line when in real_ai_mode

        self.lock = Lock()
        self._is_active = False
        
       
        

    
        
    @abstractmethod
    def is_full(self):
        """
        Returns whether there is room for additional players to join or not
        """
        pass

    @abstractmethod
    def apply_action(self, player_idx, action):
        """
        Updates the game state by applying a single (player_idx, action) tuple. Subclasses should try to override this method
        if possible
        """
        pass

    @abstractmethod
    def is_finished(self):
        """
        Returns whether the game has concluded or not
        """
        pass

    def is_ready(self):
        """
        Returns whether the game can be started. Defaults to having enough players
        """
        return self.is_full()

    @property
    def is_active(self):
        """
        Whether the game is currently being played
        """
        return self._is_active

    @property
    def reset_timeout(self):
        """
        Number of milliseconds to pause game on reset
        """
        return 3000

    def apply_actions(self):
        """
        Updates the game state by applying each of the pending actions in the buffer. Is called by the tick method. Subclasses
        should override this method if joint actions are necessary. If actions can be serialized, overriding `apply_action` is 
        preferred
        """
        for i in range(len(self.players)):
            try:
                while True:
                    action = self.pending_actions[i].get(block=False)
                    self.apply_action(i, action)
                    
            except Empty:
                pass

    def activate(self):
        """
        Activates the game to let server know real-time updates should start. Provides little functionality but useful as
        a check for debugging
        """
        self._is_active = True

    def deactivate(self):
        """
        Deactives the game such that subsequent calls to `tick` will be no-ops. Used to handle case where game ends but 
        there is still a buffer of client pings to handle
        """
        self._is_active = False

    def reset(self):
        """
        Restarts the game while keeping all active players by resetting game stats and temporarily disabling `tick`
        """
        if not self.is_active:
            raise ValueError("Inactive Games cannot be reset")
        if self.is_finished():
            #Maybe add here logfile?
            return self.Status.DONE
        self.deactivate()
        self.activate()
        return self.Status.RESET

    def needs_reset(self):
        """
        Returns whether the game should be reset on the next call to `tick`
        """
        return False

    def tick(self):
        """
        Updates the game state by applying each of the pending actions. This is done so that players cannot directly modify
        the game state, offering an additional level of safety and thread security. 

        One can think of "enqueue_action" like calling "git add" and "tick" like calling "git commit"

        Subclasses should try to override `apply_actions` if possible. Only override this method if necessary
        """
        if not self.is_active:
            return self.Status.INACTIVE
        if self.needs_reset():
            self.reset()
            return self.Status.RESET

        self.apply_actions()
        return self.Status.DONE if self.is_finished() else self.Status.ACTIVE

    def enqueue_action(self, player_id, action):
        """
        Add (player_id, action) pair to the pending action queue, without modifying underlying game state

        Note: This function IS thread safe
        """
        if not self.is_active:
            # Could run into issues with is_active not being thread safe
            return
        if player_id not in self.players:
            # Only players actively in game are allowed to enqueue actions
            return
        try:
            player_idx = self.players.index(player_id)
            self.pending_actions[player_idx].put(action)
        except Full:
            pass

    def get_state(self):
        """
        Return a JSON compatible serialized state of the game. Note that this should be as minimalistic as possible
        as the size of the game state will be the most important factor in game performance. This is sent to the client
        every frame update.
        """
        return {"players": self.players}

    def to_json(self):
        """
        Return a JSON compatible serialized state of the game. Contains all information about the game, does not need to
        be minimalistic. This is sent to the client only once, upon game creation
        """
        return self.get_state()

    def is_empty(self):
        """
        Return whether it is safe to garbage collect this game instance
        """
        return not self.num_players

    def add_player(self, player_id, idx=None, buff_size=-1):
        """
        Add player_id to the game
        """
        if self.is_full():
            raise ValueError("Cannot add players to full game")
        if self.is_active:
            raise ValueError("Cannot add players to active games")
        if not idx and self.EMPTY in self.players:
            idx = self.players.index(self.EMPTY)
        elif not idx:
            idx = len(self.players)

        padding = max(0, idx - len(self.players) + 1)
        for _ in range(padding):
            self.players.append(self.EMPTY)
            self.pending_actions.append(self.EMPTY)

        self.players[idx] = player_id
        self.pending_actions[idx] = Queue(maxsize=buff_size)

    def add_spectator(self, spectator_id):
        """
        Add spectator_id to list of spectators for this game
        """
        if spectator_id in self.players:
            raise ValueError("Cannot spectate and play at same time")
        self.spectators.add(spectator_id)

    def remove_player(self, player_id):
        """
        Remove player_id from the game
        """
        try:
            idx = self.players.index(player_id)
            self.players[idx] = self.EMPTY
            self.pending_actions[idx] = self.EMPTY
        except ValueError:
            return False
        else:
            return True

    def remove_spectator(self, spectator_id):
        """
        Removes spectator_id if they are in list of spectators. Returns True if spectator successfully removed, False otherwise
        """
        try:
            self.spectators.remove(spectator_id)
        except ValueError:
            return False
        else:
            return True

    def clear_pending_actions(self):
        """
        Remove all queued actions for all players
        """
        for i, player in enumerate(self.players):
            if player != self.EMPTY:
                queue = self.pending_actions[i]
                queue.queue.clear()

    @property
    def num_players(self):
        return len([player for player in self.players if player != self.EMPTY])

    def get_data(self):
        """
        Return any game metadata to server driver. Really only relevant for Psiturk code
        """
        data = {}
        data["layout"] = CURR_LAYOUT
        return data


class DummyGame(Game):

    """
    Standin class used to test basic server logic
    """

    def __init__(self, **kwargs):
        super(DummyGame, self).__init__(**kwargs)
        self.counter = 0

    def is_full(self):
        return self.num_players == 2

    def apply_action(self, idx, action):
        pass

    def apply_actions(self):
        self.counter += 1

    def is_finished(self):
        return self.counter >= 100

    def get_state(self):
        state = super(DummyGame, self).get_state()
        state['count'] = self.counter
        return state


class DummyInteractiveGame(Game):

    """
    Standing class used to test interactive components of the server logic
    """

    def __init__(self, **kwargs):
        super(DummyInteractiveGame, self).__init__(**kwargs)
        self.max_players = int(kwargs.get('playerZero', 'human') ==
                               'human') + int(kwargs.get('playerOne', 'human') == 'human')
        self.max_count = kwargs.get('max_count', 30)
        self.counter = 0
        self.counts = [0] * self.max_players

    def is_full(self):
        return self.num_players == self.max_players

    def is_finished(self):
        return max(self.counts) >= self.max_count

    def apply_action(self, player_idx, action):
        if action.upper() == Direction.NORTH:
            self.counts[player_idx] += 1
        if action.upper() == Direction.SOUTH:
            self.counts[player_idx] -= 1

    def apply_actions(self):
        super(DummyInteractiveGame, self).apply_actions()
        self.counter += 1

    def get_state(self):
        state = super(DummyInteractiveGame, self).get_state()
        state['count'] = self.counter
        for i in range(self.num_players):
            state['player_{}_count'.format(i)] = self.counts[i]
        return state


class OvercookedGame(Game):
    """
    Class for bridging the gap between Overcooked_Env and the Game interface

    Instance variable:
        - max_players (int): Maximum number of players that can be in the game at once
        - mdp (OvercookedGridworld): Controls the underlying Overcooked game logic
        - score (int): Current reward acheived by all players
        - max_time (int): Number of seconds the game should last
        - npc_policies (dict): Maps user_id to policy (Agent) for each AI player
        - npc_state_queues (dict): Mapping of NPC user_ids to LIFO queues for the policy to process
        - curr_tick (int): How many times the game server has called this instance's `tick` method
        - ticker_per_ai_action (int): How many frames should pass in between NPC policy forward passes. 
            Note that this is a lower bound; if the policy is computationally expensive the actual frames
            per forward pass can be higher
        - action_to_overcooked_action (dict): Maps action names returned by client to action names used by OvercookedGridworld
            Note that this is an instance variable and not a static variable for efficiency reasons
        - human_players (set(str)): Collection of all player IDs that correspond to humans
        - npc_players (set(str)): Collection of all player IDs that correspond to AI
        - randomized (boolean): Whether the order of the layouts should be randomized

    Methods:
        - npc_policy_consumer: Background process that asynchronously computes NPC policy forward passes. One thread
            spawned for each NPC
        - _curr_game_over: Determines whether the game on the current mdp has ended
    """
    
    log_ball = []
    def __init__(self, layouts=["cramped_room_tutorial"], mdp_params={}, num_players=2, gameTime=30, playerZero='human', playerOne='human', showPotential=False, randomized=False, **kwargs):

        super(OvercookedGame, self).__init__(**kwargs)
        self.show_potential = showPotential
        self.mdp_params = mdp_params
        self.layouts = layouts
        self.max_players = int(num_players)
        self.mdp = None
        self.mp = None
        self.score = 0
        self.time_ball = 0
        self.slip = False #flag to know if astro is slipping or not
        self.start_time_ball = 0
        self.phi = 0
        self.max_time = min(int(gameTime), MAX_GAME_TIME)
        self.npc_policies = {}
        self.npc_state_queues = {}
        self.action_to_overcooked_action = {
            "STAY": Action.STAY,
            "UP": Direction.NORTH,
            "DOWN": Direction.SOUTH,
            "LEFT": Direction.WEST,
            "RIGHT": Direction.EAST,
            "SPACE": Action.INTERACT
        }
        self.ticks_per_ai_action = 4
        self.curr_tick = 0
        self.human_players = set()
        self.npc_players = set()
        self.t_ball = 0
        




        if randomized:
            random.shuffle(self.layouts)

        if playerZero != 'human':
            player_zero_id = playerZero + '_0'
            self.add_player(player_zero_id, idx=0, buff_size=1, is_human=False)
            self.npc_policies[player_zero_id] = self.get_policy(
                playerZero, idx=0)
            self.npc_state_queues[player_zero_id] = LifoQueue()

        if playerOne != 'human':
            player_one_id = playerOne + '_1'
            self.add_player(player_one_id, idx=1, buff_size=1, is_human=False)
            self.npc_policies[player_one_id] = self.get_policy(
                playerOne, idx=1)
            self.npc_state_queues[player_one_id] = LifoQueue()


    def _curr_game_over(self):
        with open(f"{PATH}/game_debug.txt", "w") as f:
            f.write(str(bool(self.state.objects)))
            f.close()
        return (not self.state.objects and not self.state.players[0].held_object)

    def needs_reset(self):
        return self._curr_game_over() and not self.is_finished()

    def add_player(self, player_id, idx=None, buff_size=-1, is_human=True):
        super(OvercookedGame, self).add_player(
            player_id, idx=idx, buff_size=buff_size)
        if is_human:
            self.human_players.add(player_id)
        else:
            self.npc_players.add(player_id)

    def remove_player(self, player_id):
        removed = super(OvercookedGame, self).remove_player(player_id)
        if removed:
            if player_id in self.human_players:
                self.human_players.remove(player_id)
            elif player_id in self.npc_players:
                self.npc_players.remove(player_id)
            else:
                raise ValueError("Inconsistent state")

    def npc_policy_consumer(self, policy_id):
        queue = self.npc_state_queues[policy_id]
        policy = self.npc_policies[policy_id]
        while self._is_active:
            state = queue.get()
            npc_action, _ = policy.action(state)
            super(OvercookedGame, self).enqueue_action(policy_id, npc_action)

    def is_full(self):
        return self.num_players >= self.max_players

    def is_finished(self):
        val = not self.layouts and self._curr_game_over()
        return val

    def is_empty(self):
        """
        Game is considered safe to scrap if there are no active players or if there are no humans (spectating or playing)
        """
        return super(OvercookedGame, self).is_empty() or not self.spectators and not self.human_players

    def is_ready(self):
        """
        Game is ready to be activated if there are a sufficient number of players and at least one human (spectator or player)
        """
        return super(OvercookedGame, self).is_ready() and not self.is_empty()

    def apply_action(self, player_id, action):
        pass

    def apply_actions(self):
    
        # Default joint action, as NPC policies and clients probably don't enqueue actions fast
        # enough to produce one at every tick
        joint_action = [Action.STAY] * len(self.players)

        # Synchronize individual player actions into a joint-action as required by overcooked logic
        for i in range(len(self.players)):
            try:
                joint_action[i] = self.pending_actions[i].get(block=False)
            except Empty:
                pass

        # Apply overcooked game logic to get state transition
        prev_state = self.state

        self.state, info = self.mdp.get_state_transition(
            prev_state, joint_action)

        if self.show_potential:
            self.phi = self.mdp.potential_function(
                prev_state, self.mp, gamma=0.99)

        # Send next state to all background consumers if needed
        if self.curr_tick % self.ticks_per_ai_action == 0:
            for npc_id in self.npc_policies:
                self.npc_state_queues[npc_id].put(self.state, block=False)

        # Update score based on soup deliveries that might have occured
        curr_reward = sum(info['sparse_reward_by_agent'])
        self.score += curr_reward

        if self.state.players[0].has_object() and self.start_time_ball == 0:  # Pick up object
            self.start_time_ball = time()
        # holding object
        elif (self.state.players[0].has_object()) and (self.start_time_ball != 0):
            add_time_ball = time() - self.start_time_ball
            self.start_time_ball = time()
            self.time_ball += add_time_ball

        elif not self.state.players[0].has_object():
            self.start_time_ball = 0

        self.slip = SLIP
        with open(f"{PATH}/SlipDebug.txt", "a") as h:
            h.write("self.slip esta a: " + str(self.slip) + "\n")
            h.close()


        # Return about the current transition
        return prev_state, joint_action, info

    def enqueue_action(self, player_id, action):
        overcooked_action = self.action_to_overcooked_action[action]
        super(OvercookedGame, self).enqueue_action(
            player_id, overcooked_action)

    def reset(self):
        status = super(OvercookedGame, self).reset()
        if status == self.Status.RESET:
            # Hacky way of making sure game timer doesn't "start" until after reset timeout has passed
            self.start_time += self.reset_timeout / 1000

    def tick(self):
        self.curr_tick += 1
        return super(OvercookedGame, self).tick()

    def activate(self):

        global CURR_LAYOUT

        super(OvercookedGame, self).activate()

        # Sanity check at start of each game
        if not self.npc_players.union(self.human_players) == set(self.players):
            raise ValueError("Inconsistent State")

        self.curr_layout = self.layouts.pop()
        with open(f"{PATH}/LayoutDebug.txt", "a") as h:
            h.write("curr_layout " + str(self.layouts) + "\n")
            h.close()
        CURR_LAYOUT = copy.copy(self.curr_layout)
        self.mdp = OvercookedGridworld.from_layout_name(
            self.curr_layout, **self.mdp_params)
        with open(f"{PATH}/LayoutDebug.txt", "a") as h:
            h.write("curr_layout" + str(self.curr_layout) + "\n")
            h.close()
        if self.show_potential:
            self.mp = MotionPlanner.from_pickle_or_compute(
                self.mdp, counter_goals=NO_COUNTERS_PARAMS)
        self.state = self.mdp.get_standard_start_state()
        if self.show_potential:
            self.phi = self.mdp.potential_function(
                self.state, self.mp, gamma=0.99)
        self.start_time = time()
        self.start_time_ball = 0
        self.slip = SLIP
        self.curr_tick = 0
        self.score = 0
        self.time_ball = 0
        self.threads = []
        for npc_policy in self.npc_policies:
            self.npc_policies[npc_policy].reset()
            self.npc_state_queues[npc_policy].put(self.state)
            t = Thread(target=self.npc_policy_consumer, args=(npc_policy,))
            self.threads.append(t)
            t.start()

    def deactivate(self):
        super(OvercookedGame, self).deactivate()
        # Ensure the background consumers do not hang
        for npc_policy in self.npc_policies:
            self.npc_state_queues[npc_policy].put(self.state)

        # Wait for all background threads to exit
        for t in self.threads:
            t.join()

        # Clear all action queues
        self.clear_pending_actions()

    def get_state(self):
        state_dict = {}
        state_dict['potential'] = self.phi if self.show_potential else None
        state_dict['state'] = self.state.to_dict()
        state_dict['score'] = self.score
        # Alteracao do tempo
        state_dict['time_left'] = time() - self.start_time
        state_dict['time_ball'] = self.time_ball
        state_dict['slip'] = self.slip
        return state_dict

    def to_json(self):
        obj_dict = {}
        obj_dict['terrain'] = self.mdp.terrain_mtx if self._is_active else None
        obj_dict['state'] = self.get_state() if self._is_active else None
        obj_dict['userid'] = self.userid if self._is_active else None
        obj_dict['layout'] = self.curr_layout if self._is_active else None
        return obj_dict

    def get_policy(self, npc_id, idx=0):
        if npc_id.lower().startswith("rllib"):
            try:
                # Loading rllib agents requires additional helpers
                fpath = os.path.join(AGENT_DIR, npc_id, 'agent')

                agent = load_agent(fpath, agent_index=idx)

                fix_bc_path(fpath)
                return agent
            except Exception as e:
                raise IOError(
                    "Error loading Rllib Agent\n{}".format(e.__repr__()))
            finally:
                # Always kill ray after loading agent, otherwise, ray will crash once process exits
                if ray.is_initialized():
                    ray.shutdown()
        else:
            try:
                fpath = os.path.join(AGENT_DIR, npc_id, 'agent.pickle')
                with open(fpath, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                raise IOError("Error loading agent\n{}".format(e.__repr__()))
    


class OvercookedPsiturk(OvercookedGame):
    """
    Wrapper on OvercookedGame that handles additional housekeeping for Psiturk experiments

    Instance Variables:
        - trajectory (list(dict)): list of state-action pairs in current trajectory
        - psiturk_uid (string): Unique id for each psiturk game instance (provided by Psiturk backend)
            Note, this is not the user id -- two users in the same game will have the same psiturk_uid
        - trial_id (string): Unique identifier for each psiturk trial, updated on each call to reset
            Note, one OvercookedPsiturk game handles multiple layouts. This is how we differentiate

    Methods:
        get_data: Returns the accumulated trajectory data and clears the self.trajectory instance variable

    """

    def __init__(self, *args, psiturk_uid='-1', **kwargs):
        super(OvercookedPsiturk, self).__init__(
            *args, showPotential=False, **kwargs)
        self.psiturk_uid = psiturk_uid
        self.trajectory = []

    def activate(self):
        """
        Resets trial ID at start of new "game"
        """
        super(OvercookedPsiturk, self).activate()
        self.trial_id = self.psiturk_uid + str(self.start_time)

    def apply_actions(self):
        """
        Applies pending actions then logs transition data
        """
        # Apply MDP logic
        prev_state, joint_action, info = super(
            OvercookedPsiturk, self).apply_actions()

        # Log data to send to psiturk client
        curr_reward = sum(info['sparse_reward_by_agent'])
        transition = {
            "state": json.dumps(prev_state.to_dict()),
            "joint_action": json.dumps(joint_action),
            "reward": curr_reward,
            # "time_left" : max(self.max_time - (time() - self.start_time), 0),
            "time_left": time() - self.start_time,
            "time_ball": self.time_ball,
            "slip": self.slip,
            "score": self.score,
            "time_elapsed": time() - self.start_time,
            "cur_gameloop": self.curr_tick,
            "layout": json.dumps(self.mdp.terrain_mtx),
            "layout_name": self.curr_layout,
            "trial_id": self.trial_id,
            "player_0_id": self.players[0],
            "player_1_id": self.players[1],
            "player_0_is_human": self.players[0] in self.human_players,
            "player_1_is_human": self.players[1] in self.human_players
        }

        self.trajectory.append(transition)

    def get_data(self):
        """
        Returns and then clears the accumulated trajectory
        """
        data = {"uid": self.psiturk_uid + "_" +
                str(time()), "trajectory": self.trajectory}
        self.trajectory = []
        return data


class OvercookedTutorial(OvercookedGame):

    """
    Wrapper on OvercookedGame that includes additional data for tutorial mechanics, most notably the introduction of tutorial "phases"

    Instance Variables:
        - curr_phase (int): Indicates what tutorial phase we are currently on
        - phase_two_score (float): The exact sparse reward the user must obtain to advance past phase 2
    """

    def __init__(self, layouts=["tutorial_0"], mdp_params={}, playerZero='human', playerOne='AI', phaseTwoScore=15, **kwargs):
        super(OvercookedTutorial, self).__init__(layouts=layouts, mdp_params=mdp_params,
                                                 playerZero=playerZero, playerOne=playerOne, showPotential=False, **kwargs)
        self.phase_two_score = phaseTwoScore
        self.phase_two_finished = False
        self.max_time = 0
        self.max_players = 2
        self.ticks_per_ai_action = 8
        self.curr_phase = 0

    @property
    def reset_timeout(self):
        return 1

    def needs_reset(self):
        if self.curr_phase == 0:
            return self.score > 0
        elif self.curr_phase == 1:
            return self.score > 0
        elif self.curr_phase == 2:
            return self.phase_two_finished
        return False

    def is_finished(self):
        return not self.layouts and self.score >= float('inf')

    def reset(self):
        super(OvercookedTutorial, self).reset()
        self.curr_phase += 1

    def get_policy(self, *args, **kwargs):
        return TutorialAI()

    def apply_actions(self):
        """
        Apply regular MDP logic with retroactive score adjustment tutorial purposes
        """
        _, _, info = super(OvercookedTutorial, self).apply_actions()

        human_reward, ai_reward = info['sparse_reward_by_agent']

        # We only want to keep track of the human's score in the tutorial
        self.score -= ai_reward

        # Phase two requires a specific reward to complete
        if self.curr_phase == 2:
            self.score = 0
            if human_reward == self.phase_two_score:
                self.phase_two_finished = True


class DummyOvercookedGame(OvercookedGame):
    """
    Class that hardcodes the AI to be random. Used for debugging
    """

    def __init__(self, layouts=["cramped_room_tutorial"], **kwargs):
        super(DummyOvercookedGame, self).__init__(layouts, **kwargs)


    def get_policy(self, *args, **kwargs):
        
        return DummyAI()


class DummyAI():
    """
    Astro tutorial behaviour
    """
    
    
    ASTRO_POS = (0,0)
    action_dict = {
        "stay": Action.STAY,
        "up": Direction.NORTH,
        "down": Direction.SOUTH,
        "left": Direction.WEST,
        "right": Direction.EAST
    }
    
    OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    NUM_ROWS = 15
    NUM_COLUMNS = 15

    NORTH = (0, -1)  #0
    SOUTH = (0, 1)  #1
    EAST = (1, 0)   #2
    WEST = (-1, 0)  #3
    mdp_ind = []
    p_join = []

    INIT = True
   
        
    
    TARGET = [False, False]
    
    ACTION_SPACE = tuple(range(len(ACTION_MEANING)))
    JOINT_ACTION_SPACE = list(itertools.product(ACTION_SPACE, repeat=2))
    dist = np.zeros(4) #distancia as onions
       
    last_action = 0
    t = 0
    def check_index(self, state):

        for ind, val in enumerate(self.mdp_ind):
            if np.array_equal(np.array(val),np.array(state)):
                return ind
            
    def prime_action_choice(self, prob):
        # Bias favouring robot actions == stay or act
        
        aux = copy.copy(prob)

        for i in range(len(prob)):
            if i >=24 and aux[i] != 0:
                aux[i] = aux[i]*1.1

        prob = aux / sum(aux)
        
        
        #print("prob after: ", prob)
        #ret = np.random.choice(range(len(self.JOINT_ACTION_SPACE)), p=prob)
        ret = np.argmax(prob)
        return ret



    def action_tutorial(self, state): #Astro Handcoded Behavior
        global SLIP 
        global CURR_CONDITION
        global CURR_LAYOUT
        SLIP = False 
        #self.env.state[8] = 4 pan
        a0_row = state.players[0].position[0]
        a0_column = state.players[0].position[1]  #a0 human
        a1_row = state.players[1].position[0] #a1 - astro
        a1_column = state.players[1].position[1]
        
        a0_hand = state.players[0].held_object 

        #onions = self.env.onions
        astro_pos = (a1_row, a1_column)
        self.ASTRO_POS = (astro_pos[1],astro_pos[0])
        dist = np.zeros(3) #3 onions

        onions_list = list(state.objects.values()) #Only has the balls in the counters
        aux_list = [False, False, False] #Helps to see what balls we have calculated the distance

        #onions distance to astro
        for i in range(0, len(onions_list)):  #onions com status 0 (bancada)
            index = onions_list[i].index - 1
            aux_list[index] = True
            dist[index] = np.linalg.norm(np.array([onions_list[i].position[0], onions_list[i].position[1]])- np.array([a1_row,a1_column]))
                    #distancias sao sempre nmrs grandes.. ?     
            

        for i in range(len(aux_list)): #outras onions (status 1 e 2)
            if aux_list[i] == False:
                dist[i] = 100 
                     
        
        
        #if robot
        go_to = None
        if a0_hand is not None: #human holding ball
            action = self._action_to_move_to(state, (a0_column, a0_row)) #go to humans position

        else: #go to neareast onion pos
            min_ball = np.argmin(dist) #0, 1 or 2
            for ball in onions_list:
                if ball.index - 1 == min_ball:
                    go_to = (ball.position[1],ball.position[0])
            if go_to is None:
                go_to = (a0_column, a0_row) 
            action = self._action_to_move_to(state, go_to)

        ice_pos = self.find_ice_pos()

        if CURR_LAYOUT == "cramped_room_tutorial":
            if CURR_CONDITION == "Low":
                self.S_COEF = 0.65
            elif CURR_CONDITION == "High":
                self.S_COEF = 0.15
            elif CURR_CONDITION == "Human":
                self.S_COEF = 0.15
    
        if random.random()<self.S_COEF and (self.ASTRO_POS in ice_pos):
            next_move = self.slip_move(self.ASTRO_POS, state)
            action = self._action_to_move_to(state,next_move) 

        return action
    
    def go_to_node(self, node, state_env):
    
        #if self.index == 1: # ROBOT
        target = (self.AGENT_POS[node][1], self.AGENT_POS[node][0]) 
        #action_env = self._action_to_move_to(state_env, self.AGENT_POS[node], is_teammate_obstacle=False)
        action_env = self._action_to_move_to(state_env, target, is_teammate_obstacle=False)
        
        return action_env
    
    def find_node(self, ad_matrix, state_mdp, action_mdp):

        adjacencies = np.where(ad_matrix[state_mdp[0]] == 1)[0]
        downgrade_to_lower_index = int(action_mdp) >= len(adjacencies)
        action_mdp = 0 if downgrade_to_lower_index else action_mdp
        node = adjacencies[action_mdp]
        return node

    def action_converter(self, state, state_mdp, action_mdp):
        with open(f"{PATH}/DebugAstro.txt", "a") as h:
                h.write("action_mdp: " + str(ACTION_MEANING[action_mdp]) + "\n")
                h.close() 
        global SLIP 
        SLIP = False 
    
        pos = (self.ASTRO_POS[1], self.ASTRO_POS[0])
        
        #if human is holding onion
        if state.players[0].held_object is not None:
            pos = (self.ASTRO_POS[1], self.ASTRO_POS[0])
            ice_pos = self.find_ice_pos() #pos do OVERCOOKED-Gym (y, -x) ---> (demo: (x, -y))
            n = random.random()
            
            if n < self.S_COEF and (self.ASTRO_POS in ice_pos):
                next_move = self.slip_move(self.ASTRO_POS, state)
                return self._action_to_move_to(state, (next_move[0],next_move[1])) 
            
    
            human_pos = (state.players[0].position[1], state.players[0].position[0])

            with open(f"{PATH}/PolicyDebug.txt", "a") as h:
                        h.write("go to human state\n")
                        h.close()
            return self._action_to_move_to(state, human_pos) #Go to humans position""" 
        
       
        with open(f"{PATH}/PolicyDebug.txt", "a") as h:
                        h.write("cond: " + str(pos) + str(self.AGENT_POS) + "\n")
                        h.close()
        # IF STAY
        if action_mdp == ACTION_MEANING.index("stay"):
        
            #if human pos is in self.human_pos or self.target[0]
            #then self.target[0] = True
            #elif not return go to node. --> Implement human behavior?
    
            #if robot:
            pos = (self.ASTRO_POS[1], self.ASTRO_POS[0]) 

            
            if pos in self.AGENT_POS or self.TARGET[1]:
                    self.TARGET[1] = True
                   
                    return Action.STAY
            elif not self.TARGET[1]:
                pos = (self.ASTRO_POS[1], self.ASTRO_POS[0])
                ice_pos = self.find_ice_pos()
                if random.random()<self.S_COEF and (self.ASTRO_POS in ice_pos):
                        next_move = self.slip_move(self.ASTRO_POS,state)
                        return self._action_to_move_to(state,(next_move[0], next_move[1]))
                


                with open(f"{PATH}/PolicyDebug.txt", "a") as h:
                        h.write("IF STAY: go to target node\n")
                        h.close()
                return self.go_to_node(state_mdp[0], state) #go to astro target/node
                #return self.action_dict[action]

            #return Action.STAY

        # IF ACT 
        elif action_mdp == ACTION_MEANING.index("act"):
            
            #return Action.INTERACT
            #if robot
            pos = (self.ASTRO_POS[1], self.ASTRO_POS[0])

            if pos in self.AGENT_POS  or self.TARGET[1]:
                self.TARGET[1]= True
                with open(f"{PATH}/PolicyDebug.txt", "a") as h:
                        h.write("IF ACT: stay \n")
                        h.close()
                
                return Action.STAY
            elif not self.TARGET[1]: 
                ice_pos = self.find_ice_pos()
                if random.random()<self.S_COEF and (self.ASTRO_POS in ice_pos):
                    next_move = self.slip_move(self.ASTRO_POS,state)
                    return self._action_to_move_to(state,next_move) 
                

                with open(f"{PATH}/PolicyDebug.txt", "a") as h:
                        h.write("IF ACT: go to target node\n")
                        h.close()
                return self.go_to_node(state_mdp[0], state)
                #return self.action_dict[action]
            #elif human? do nothing?
        # IF MOVE
        else: 
            if CURR_LAYOUT == "level_one":
                ADJACENCY_MATRIX = np.array(
                    [
                        [0, 1, 0, 0, 0, 1, 0, 0],
                        [1, 0, 1, 1, 0, 0, 0, 0],
                        [0, 1, 0, 1, 0, 0, 0, 0],
                        [0, 1, 1, 0, 1, 1, 0, 0],
                        [0, 0, 0, 1, 0, 0, 1, 1],
                        [1, 0, 0, 1, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1, 1, 0, 1],
                        [0, 0, 0, 0, 1, 0, 1, 0],
                    ]
                )
            elif CURR_LAYOUT == "level_two":

                #LAB2 AD_MAT
                ADJACENCY_MATRIX = np.array(
                    [
                        [0, 0, 1, 0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0, 0, 0, 0],
                        [1, 1, 0, 1, 0, 0, 0, 0, 1],
                        [0, 1, 1, 0, 0, 0, 0, 1, 0],
                        [1, 0, 0, 0, 0, 1, 0, 0, 1],
                        [0, 0, 0, 0, 1, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0, 0, 1, 0, 1],
                        [0, 0, 1, 0, 1, 1, 0, 1, 0],

                    ]
                )
             
            
            #if human - nothing
            # if robot:      
            pos = (self.ASTRO_POS[1], self.ASTRO_POS[0])

            onions_list = list(state.objects.values()) #Only has the balls in the counters
            aux_list = [False, False, False, False] #Helps to see what balls we have calculated the distance

            #onions distance to astro
            for i in range(0, len(onions_list)):  #Ao introduzir condição que coloca a 100 a dist qd o status n e 0 (n esta em cima da mesa), o status das bolas estranhamente nunca mudam.

                index = onions_list[i].index - 1
                aux_list[index] = True
                self.dist[index] = np.linalg.norm(np.array([self.ASTRO_POS[0],self.ASTRO_POS[1]])-np.array([onions_list[i].position[0], onions_list[i].position[1]])) 
                    #distancias sao sempre nmrs grandes.. ?     
            
            
            if state.players[0].held_object is not None:
                onion = state.players[0].held_object 
                self.dist[onion.index-1] = np.linalg.norm(np.array([self.ASTRO_POS[0],self.ASTRO_POS[1]])-np.array([onion.position[0], onion.position[1]]))
                aux_list[onion.index-1] = True

            for i in range(len(aux_list)):
                if aux_list[i] == False:
                    self.dist[i] = 100 #FIXME Manter isto por agora, alterar mais tarde
                     

            
            with open(f"{PATH}/DebugAstro.txt", "a") as h:
                h.write("dist: " + str(self.dist) + "\n")
                h.close() 

            
            ice_pos = self.find_ice_pos()
            if random.random()<self.S_COEF and (self.ASTRO_POS in ice_pos):
                    next_move = self.slip_move(self.ASTRO_POS,state)
                    with open(f"{PATH}/PolicyDebug.txt", "a") as h:
                        h.write("IF MOVE: slip\n")
                        h.close()
                
                    return self._action_to_move_to(state,next_move) 
            
            with open(f"{PATH}/AllDebug.txt", "a") as h:
                        h.write("IF MOVE cond: \npos: " + str(pos)+  "\nAGENT_POS: "+ str(self.AGENT_POS) + "\nTARGET1: " + str(self.TARGET[1]) + "\n min dist: " + str(min(self.dist)) + " \n")
                        h.close()
            if pos in self.AGENT_POS or self.TARGET[1] or min(self.dist) <= 1:
                self.TARGET[1] = True #flag so fica true qd chega ao centro do no (pos especifica) -> Assim que sai do node, target fica falso
                node = self.find_node(ADJACENCY_MATRIX,state_mdp, action_mdp)
                #target = self.AGENT_POS[node]
                with open(f"{PATH}/AllDebug.txt", "a") as h:
                        h.write("IF MOVE: go to new node: " +str(node) +"\n")
                        h.close()
                return self.go_to_node(node,state)
                #return self.action_dict[action]
            else: #Go to target node of mdp
                with open(f"{PATH}/AllDebug.txt", "a") as h:
                        h.write("IF MOVE: go to target node (mdp)\n")
                        h.close()
                return self.go_to_node(state_mdp[0], state)  
    
    def state_converter(self, state):
        if CURR_LAYOUT == "level_one":
            STATE_MAP = np.array([
                ["2", "2", "2", "2", "2", "2", "2", "4", "7", "7", "7", "7", "7", "7", "7"],
                ["2", "2", "2", "2", "2", "2", "2", "4", "7", "7", "7", "7", "7", "7", "7"],
                ["2", "2", "2", "2", "2", "2", "2", "4", "4", "4", "7", "7", "7", "7", "7"],
                ["2", "2", "2", "2", "2", "2", "2", "4", "4", "4", "7", "7", "7", "7", "7"],
                ["2", "2", "2", "2", "2", "3", "3", "4", "4", "4", "7", "7", "7", "7", "7"],
                ["2", "2", "2", "2", "2", "3", "3", "4", "4", "4", "4", "4", "7", "7", "7"],
                ["1", "1", "1", "1", "3", "3", "3", "4", "4", "4", "4", "4", "4", "7", "7"],
                ["1", "1", "1", "1", "3", "3", "3", "4", "4", "4", "4", "4", "4", "7", "7"],
                ["1", "1", "1", "1", "3", "3", "3", "4", "4", "4", "4", "4", "4", "7", "7"],
                ["1", "1", "1", "1", "3", "3", "3", "4", "4", "6", "4", "4", "6", "6", "6"],
                ["1", "1", "1", "1", "1", "3", "3", "3", "4", "6", "6", "6", "6", "6", "6"],
                ["0", "0", "0", "0", "0", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5"],
                ["0", "0", "0", "0", "0", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5"],
                ["0", "0", "0", "0", "0", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5"],
                ["0", "0", "0", "0", "0", "5", "5", "5", "5", "5", "5", "5", "5", "5", "5"]])
        elif CURR_LAYOUT == "level_two":
            #LAB2 STATE MAP
            STATE_MAP = np.array([
                ["5", "5", "5", "5", "5", "5", "5", "6", "6", "6", "6", "7", "7", "7", "7"],
                ["5", "5", "5", "5", "5", "5", "5", "6", "6", "6", "6", "7", "7", "7", "7"],
                ["5", "5", "5", "5", "5", "5", "5", "6", "6", "6", "6", "7", "7", "7", "7"],
                ["5", "5", "5", "5", "5", "5", "5", "6", "6", "6", "6", "7", "7", "7", "7"],
                ["5", "5", "5", "5", "5", "5", "5", "6", "6", "6", "6", "7", "7", "7", "7"],
                ["5", "5", "5", "5", "5", "5", "5", "6", "6", "6", "6", "7", "7", "7", "7"],
                ["5", "5", "5", "5", "5", "5", "8", "8", "8", "8", "8", "7", "7", "7", "7"],
                ["4", "4", "4", "4", "4", "4", "8", "8", "8", "8", "3", "3", "3", "3", "3"],
                ["4", "4", "4", "4", "4", "4", "8", "8", "8", "8", "3", "3", "3", "3", "3"],
                ["4", "4", "4", "4", "4", "4", "2", "2", "2", "2", "3", "3", "3", "3", "3"],
                ["0", "0", "2", "2", "2", "4", "2", "2", "2", "2", "3", "3", "3", "3", "3"],
                ["0", "0", "0", "0", "2", "2", "2", "1", "1", "1", "3", "3", "3", "3", "3"],
                ["0", "0", "0", "0", "1", "1", "1", "1", "1", "1", "1", "3", "3", "3", "3"],
                ["0", "0", "0", "0", "1", "1", "1", "1", "1", "1", "1", "3", "3", "3", "3"],
                ["0", "0", "0", "0", "1", "1", "1", "1", "1", "1", "1", "3", "3", "3", "3"]])
        
        p1 = int(STATE_MAP[state.players[1].position[1],
                state.players[1].position[0]])  # Astro pos
        
        
        p2 = int(STATE_MAP[state.players[0].position[1],
                 state.players[0].position[0]])  # Human pos
        
        #Logica do que percebi: as onions tem 3 status:
            #0 = esta no balcao --> aparece na lista de objects
            #1 = esta nas maos to humano --> nao aparece na lista, mas aparece no holding do humano
            #2 = esta dentro do astro --> nao aparece na lista // aparece dentro do astro 1 so?
           
        #tentativa de logica: percorrer lista de objects e comparar com os 4 indices -> se estao na lista, status e 0 (em principio)
        #se nao estao na lista: 
            #holding onion do humano -> status e 1
            #em nenhum deles, esta dentro do astro -> status e 2


        all_status = [2, 2, 2, 2]

        
        onions_list = list(state.objects.values()) 
        for onion in onions_list: 
            all_status[onion.index-1] = onion.status #deve ser 0

        
    
        if state.players[0].held_object is not None:
            onion = state.players[0].held_object 
            all_status[onion.index-1] = onion.status #deve ser 1  
        
        

        
        state_mdp = [p1, p2, all_status[0], all_status[1], all_status[2], all_status[3]]  
        

        #state_mdp = [p1, p2, 0, 0, 0, 0]
        return state_mdp
    
    def action_one(self, state):
        self.S_COEF = 0.65 #Level 1 Coef
        self.ASTRO_POS = (state.players[1].position[1],
                 state.players[1].position[0])
        
        
        state_mdp = self.state_converter(state)

        with open(f"{PATH}/AllDebug.txt", "a") as h:
                        h.write("last state versus now state: " + str(self.last_state[0])+ " vs " + str(state_mdp[0])+ "\n")
                        h.close()
        if self.last_state[0] != state_mdp[0]: #Robot
            self.TARGET[1] = False  
            with open(f"{PATH}/AllDebug.txt", "a") as h:
                        h.write("I am puting target to false\n")
                        h.close()
        
        #Humano -> N importa

        if self.last_state == state_mdp:
            a_joint = self.last_action
        else:
            index = self.check_index(state_mdp)  #encontrar estado onde estou
            pi_join_state = self.p_join[index]  #politica para esse estado
            a_joint = self.JOINT_ACTION_SPACE[self.prime_action_choice(pi_join_state)]

        self.last_state=copy.copy(state_mdp)
        self.last_action = copy.copy(a_joint)

        #action_mdp = ACTION_MEANING.index(ACTION_MEANING[random.randint(0,len(ACTION_MEANING)-1)])
        action_mdp = ACTION_MEANING.index(ACTION_MEANING[a_joint[0]])  
        

        #action = self.action_converter(
        #    state, self.state_converter(state), action_mdp)
        

        action = self.action_converter(state, state_mdp, action_mdp)
        with open(f"{PATH}/PolicyDebug.txt", "a") as h:
            h.write("ACTION MDP: " + str(action_mdp) + " STATE MDP: " +
                    str(state_mdp) + " STATE ENV: " + str(state)+ " ACTION ENV: " + str(action)+ "\n")
            h.close()

        with open(f"{PATH}/AllDebug.txt", "a") as h:
            h.write("ACTION MDP: " + str(ACTION_MEANING[action_mdp]) + "STATE_MDP:" + str(state_mdp) + "\n")
            h.close()
       

        return action
    
    def action_two(self, state):
        self.S_COEF = 0.50 #Level 1 Coef
        self.ASTRO_POS = (state.players[1].position[1],
                 state.players[1].position[0])
        
        
        state_mdp = self.state_converter(state)

        with open(f"{PATH}/AllDebug.txt", "a") as h:
                        h.write("last state versus now state: " + str(self.last_state[0])+ " vs " + str(state_mdp[0])+ "\n")
                        h.close()
        if self.last_state[0] != state_mdp[0]: #Robot
            self.TARGET[1] = False  
            with open(f"{PATH}/AllDebug.txt", "a") as h:
                        h.write("I am puting target to false\n")
                        h.close()
        
        #Humano -> N importa

        if self.last_state == state_mdp:
            a_joint = self.last_action
        else:
            index = self.check_index(state_mdp)  #encontrar estado onde estou
            pi_join_state = self.p_join[index]  #politica para esse estado
            a_joint = self.JOINT_ACTION_SPACE[self.prime_action_choice(pi_join_state)]

        self.last_state=copy.copy(state_mdp)
        self.last_action = copy.copy(a_joint)

        #action_mdp = ACTION_MEANING.index(ACTION_MEANING[random.randint(0,len(ACTION_MEANING)-1)])
        action_mdp = ACTION_MEANING.index(ACTION_MEANING[a_joint[0]])  
        

        #action = self.action_converter(
        #    state, self.state_converter(state), action_mdp)
        

        action = self.action_converter(state, state_mdp, action_mdp)
        with open(f"{PATH}/PolicyDebug.txt", "a") as h:
            h.write("ACTION MDP: " + str(action_mdp) + " STATE MDP: " +
                    str(state_mdp) + " STATE ENV: " + str(state)+ " ACTION ENV: " + str(action)+ "\n")
            h.close()

        with open(f"{PATH}/AllDebug.txt", "a") as h:
            h.write("ACTION MDP: " + str(ACTION_MEANING[action_mdp]) + "STATE_MDP:" + str(state_mdp) + "\n")
            h.close()
       

        return action
    
    
    def action(self, state): #Main function
        if CURR_LAYOUT == "cramped_room_tutorial":
            action = self.action_tutorial(state)
        
        elif CURR_LAYOUT == "level_one":
            if self.INIT: #initialize variables only once
                self.INIT = False
                self.AGENT_POS = [(3, 13), (2, 7), (3, 1), (5, 7),(8, 10), (8, 12), (9, 10), (10, 4)]
                self.last_state = [10,10,10,10,10,10]
        
                try:
                    fpath = os.path.join("./static/assets/agents",'RandAI', 'mdp_ind_Lab1.npy')
                    with open(fpath, 'rb') as f:
                        self.mdp_ind = np.load(f)

                    
                except Exception as e:
                    raise IOError("Error loading agent\n{}".format(e.__repr__()))
                
                try:
                    fpath = os.path.join("./static/assets/agents",'RandAI', 'policy_Lab1.npy')
                    with open(fpath, 'rb') as f:
                        self.p_join = np.load(f)
                except Exception as e:
                    raise IOError("Error loading agent\n{}".format(e.__repr__()))
                
            action = self.action_one(state)
        elif CURR_LAYOUT == "level_two":
            if self.INIT:
                self.INIT = False
                self.AGENT_POS = [(2, 12), (9, 12), (9, 10), (10, 11),(1, 8), (5,3), (10, 1), (12, 6), (7,6)]
                self.last_state = [10,10,10,10,10,10]
                try:
                    fpath = os.path.join("./static/assets/agents",'RandAI', 'mdp_ind_Lab2.npy')
                    with open(fpath, 'rb') as f:
                        self.mdp_ind = np.load(f)

                    
                except Exception as e:
                    raise IOError("Error loading agent\n{}".format(e.__repr__()))
                
                try:
                    fpath = os.path.join("./static/assets/agents",'RandAI', 'policy_Lab2.npy')
                    with open(fpath, 'rb') as f:
                        self.p_join = np.load(f)
                except Exception as e:
                    raise IOError("Error loading agent\n{}".format(e.__repr__()))
                
            action = self.action_two(state)

        return action, None


    def find_ice_pos(self):
        base_layout_params = read_layout_dict(CURR_LAYOUT)   
        grid = base_layout_params["grid"]
            
        grid = [layout_row.strip() for layout_row in grid.split("\n")]

        grid = [[c for c in row] for row in grid]

        #Finds the positions of "I" in the layout
        ice_pos = [(i, j) for i, row in enumerate(grid) for j, val in enumerate(row) if val == "I"]
        return ice_pos
    

    def slip_move(self, pos, state_env):
        global SLIP
        SLIP = True
        #self.env.state[8] = 1 o que e isto? e o pan? 
        astro_heading = state_env.players[1].orientation
        NORTH = (0, -1)
        SOUTH = (0, 1)
        EAST  = (1, 0)
        WEST  = (-1, 0)

      #  "NORTH, SOUTH, WEST, EAST = range(4)  "

        if astro_heading == NORTH:
            direction = 0
        elif astro_heading == SOUTH:
            direction = 1
        elif astro_heading == EAST:
            direction = 2
        elif astro_heading == WEST:
            direction = 3

        x,y =  self.cell_facing_agent(pos[0], pos[1], direction )  #FIXME Implementar
        off = copy.copy(self.OFFSETS)
        off.pop(direction)
        i,j = off[np.random.choice(range(len(off)))]
        next_move = (x+i,y+j)
        with open(f"{PATH}/IceDebug.txt", "a") as h:
            h.write(" Slip!!! \n")
            h.close()

        return next_move

    def cell_facing_agent(self, row, column, direction):
        
        dr, dc = self.OFFSETS[direction]
               

        object_row = row + dr
        object_column = column + dc

        if object_row < 0: object_row = 0
        if object_row > self.NUM_ROWS: object_row = self.NUM_ROWS - 1

     

        if object_column < 0: object_column = 0
        if object_column > self.NUM_COLUMNS: object_column = self.NUM_COLUMNS - 1  

        return object_row, object_column
    
    def _action_to_move_to(self, state, target, alternative_target=None, is_teammate_obstacle=True):
        """ Returns the action index that allows the agent to get closer to the target. If the agent is already by the
        target, the action will be to turn arround"""
        
            
        base_layout_params = read_layout_dict(CURR_LAYOUT)

        grid = base_layout_params["grid"]
            
        grid = [layout_row.strip() for layout_row in grid.split("\n")]

        grid = [[c for c in row] for row in grid]

        #Finds the positions of "X" in the layout
        obstacles = [(i, j) for i, row in enumerate(grid) for j, val in enumerate(row) if val == "X"]
        
        action_meaning = A_star_search_overcooked_wrapper(self.ASTRO_POS, target, set(obstacles),len(grid),len(grid[0]))



        a1_heading =  state.players[1].orientation

       
        player_heading = (a1_heading[1], a1_heading[0])  #Trocar coordenadas
    

        # since the action stay does not equivalent in the overcooked interface, we simply make the agent change the
        # direction it is facing
        if action_meaning == "stay":
            if alternative_target is not None:
                return self._action_to_move_to(state, alternative_target)
            elif player_heading == Direction.NORTH:
                #return ACTION_MEANING.index("down")
                return self.action_dict["down"]
            else:
                #return ACTION_MEANING.index("up")
                return self.action_dict["up"]
        else:
            #return ACTION_MEANING.index(action_meaning)
            return self.action_dict[action_meaning]
    


    def reset(self):
        pass


class DummyComputeAI(DummyAI):
    """
    Performs simulated compute before randomly sampling actions. Used for debugging
    """

    def __init__(self, compute_unit_iters=1e5):
        """
        compute_unit_iters (int): Number of for loop cycles in one "unit" of compute. Number of 
                                    units performed each time is randomly sampled
        """
        super(DummyComputeAI, self).__init__()
        self.compute_unit_iters = int(compute_unit_iters)

    def action(self, state):
        # Randomly sample amount of time to busy wait
        #iters = random.randint(1, 10) * self.compute_unit_iters
        iters = 4 * self.compute_unit_iters
        # Actually compute something (can't sleep) to avoid scheduling optimizations
        val = 0
        for i in range(iters):
            # Avoid branch prediction optimizations
            if i % 2 == 0:
                val += 1
            else:
                val += 2

        # Return randomly sampled action
        return super(DummyComputeAI, self).action(state)


class StayAI():
    """
    Always returns "stay" action. Used for debugging
    """

    def action(self, state):
        return Action.STAY, None

    def reset(self):
        pass


class TutorialAI():

    COOK_SOUP_LOOP = [
        # Grab first onion
        Direction.WEST,
        Direction.WEST,
        Direction.WEST,
        Action.INTERACT,

        # Place onion in pot
        Direction.EAST,
        Direction.NORTH,
        Action.INTERACT,

        # Grab second onion
        Direction.WEST,
        Action.INTERACT,

        # Place onion in pot
        Direction.EAST,
        Direction.NORTH,
        Action.INTERACT,

        # Grab third onion
        Direction.WEST,
        Action.INTERACT,

        # Place onion in pot
        Direction.EAST,
        Direction.NORTH,
        Action.INTERACT,

        # Cook soup
        Action.INTERACT,

        # Grab plate
        Direction.EAST,
        Direction.SOUTH,
        Action.INTERACT,
        Direction.WEST,
        Direction.NORTH,

        # Deliver soup
        Action.INTERACT,
        Direction.EAST,
        Direction.EAST,
        Direction.EAST,
        Action.INTERACT,
        Direction.WEST
    ]

    COOK_SOUP_COOP_LOOP = [
        # Grab first onion
        Direction.WEST,
        Direction.WEST,
        Direction.WEST,
        Action.INTERACT,

        # Place onion in pot
        Direction.EAST,
        Direction.SOUTH,
        Action.INTERACT,

        # Move to start so this loops
        Direction.EAST,
        Direction.EAST,

        # Pause to make cooperation more real time
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY
    ]

    def __init__(self):
        self.curr_phase = -1
        self.curr_tick = -1

    def action(self, state):
        self.curr_tick += 1
        if self.curr_phase == 0:
            return self.COOK_SOUP_LOOP[self.curr_tick % len(self.COOK_SOUP_LOOP)], None
        elif self.curr_phase == 2:
            return self.COOK_SOUP_COOP_LOOP[self.curr_tick % len(self.COOK_SOUP_COOP_LOOP)], None
        return Action.STAY, None

    def reset(self):
        self.curr_tick = -1
        self.curr_phase += 1
