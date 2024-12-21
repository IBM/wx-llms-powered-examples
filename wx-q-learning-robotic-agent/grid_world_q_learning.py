#
# Copyright IBM Corp. 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# Author: Nguyen, Hung (Howie) Sy
#

import constants as Constants
from typing import List, Dict, Tuple, Union, override, Literal, Optional
from dataclasses import dataclass
from gymnasium import Env, spaces
import json
from langchain_core.prompts import PromptTemplate
import traceback
import random
import prompts
import numpy as np

@dataclass
class Step:
    """Step encapsulates state-action-reward-next-state information.
    A step refers to a single interaction between the agent and the environment. 
    It represents one iteration in the process of the agent taking an action,
    observing the consequences, and updating its knowledge (e.g., Q-values)"""
    state: Tuple[int, int]
    action: str
    reward: float
    next_state: Tuple[int, int]
    step_index: int

    # for monitoring purposes
    previous_q_value: float = None
    current_q_value: float = None

@dataclass
class Trajectory:
    steps: List[Step]

# Define the custom GridWorld environment
class GridWorldEnv(Env):
    def __init__(self, grid_size=Constants.GRID_SIZE, goal=Constants.GOAL, obstacles=Constants.OBSTACLES):
        super(GridWorldEnv, self).__init__()
        self.grid_size = grid_size
        
        # A State is a specific situation in which the agent can be in.
        # In this case, state is the cell location that is currently being occupied by the robot
        self.state = (0, 0) 
        
        self.goal = goal
        self.obstacles = obstacles
        self.action_space = spaces.Discrete(4)  # 4 actions: up, down, left, right
        self.action_name_mapping = { 0: "up", 1: "down", 2: "left", 3: "right" }

        # statistics
        self.obstacle_hits = {obs: 0 for obs in self.obstacles} # track obstacle visits
        self.wall_hits = 0 

        # although observation_space is not directly used in this code, 
        # but it is to comply with the OpenAI Gym API standards
        self.observation_space = spaces.Tuple((
            spaces.Discrete(grid_size[0]),
            spaces.Discrete(grid_size[1])
        ))

    @override
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = (0, 0)
        self.obstacle_hits = {obs: 0 for obs in self.obstacles} # track obstacle visits
        self.wall_hits = 0 
        return self.state, {}

    @override
    def step(self, action):
        """ From the Markov Decision Process (MDP) and RL perspective, this method can be seen 
        as an implementation example of a deterministic transition model of the environment.
        Mathematically, it is expressed as P(s'|s,a)=1). In other words, for a given state s and action a, the next state s' uniquely determined.
        It realizes the transition dynamics of the environment, defining how the environment responds to actions in a deterministic manner.
        """
        
        x, y = self.state
        next_x, next_y = x, y  # Start with the current position
        done = False

        # Calculate potential next state based on the action
        if action == 0 and x > 0:  # Up
            next_x -= 1
        elif action == 1 and x < self.grid_size[0] - 1:  # Down
            next_x += 1
        elif action == 2 and y > 0:  # Left
            next_y -= 1
        elif action == 3 and y < self.grid_size[1] - 1:  # Right
            next_y += 1
        
        next_state = (next_x, next_y)

        # if the agent hits one of the walls, then it is not allowed and 
        # therefore as a result we will have next_state becomes self.state, i.e staying in the same state
        if next_state == self.state: # a wall hit
            self.wall_hits += 1
            reward = -15  # Penalty for invalid action - hitting the wall
        elif next_state == self.goal: # reaching the goal
            reward = 30 
            done = True
        elif next_state in self.obstacles:  # move in an obstacle
            reward = -15  # Penalty for hitting an obstacle
            self.obstacle_hits[next_state] += 1  # Increment visit counter
            next_state = self.state  # Stay in the same state as moving into obstacle states is not allowed
        else:
            # Update state and calculate reward
            reward = -1  # Default step cost

        self.state = next_state

        # to comply the gym specification, the following returned: observation, reward, terminated, truncated, {}
        return self.state, reward, done, False, {}

class QLearningEngine:
    """This class is responsible for the Q-Learning logic and the learning algorithm, 
    separating it from the environment and agent"""
    def __init__(self, env: GridWorldEnv):
        self.env = env
        self.q_table = np.zeros(env.grid_size + (env.action_space.n,))

    def reset(self):
        self.q_table = np.zeros(self.env.grid_size + (self.env.action_space.n,))

    def choose_action(self, state, epsilon):
        return self.choose_action_using_epsilon_greedy_policy(state, epsilon)
    
    def choose_action_using_llm_epsilon_greedy_policy(self, step_index, state, epsilon, movement_log: List[str], llm):
        """ (experimental) """
        q_table_needs_to_be_updated = False
        getting_stuck_likelihood = -1

        q_state = self.q_table[state]

        if step_index > 0 and step_index % 5 == 0: # skip for the first step and check every 5 steps
            print("\n\n*LLM reviews the current movement log to determine the likelihood of getting stuck...")
            llm_response = self.review_the_movement_log(current_state=state, 
                                                        q_state=q_state,
                                                        movement_log=movement_log, llm=llm)
            print(f"\n*LLM's response: {llm_response}")

            llm_error = llm_response.get("error", None)
            if llm_error:
                getting_stuck_likelihood = -1
                print(f"\n*Warning: Error when querying the LLM: {llm_error}, getting_stuck_likelihood is set to -1")
            else:
                try:
                    getting_stuck_likelihood = float(llm_response.get("getting_stuck_likelihood", -1))
                except Exception as error:
                    getting_stuck_likelihood = -1
                    print(f"\n*Warning: Error when reading getting_stuck_likelihood: {error}, getting_stuck_likelihood is set to -1")
            
        THRESHOLD = 80.0
        if getting_stuck_likelihood >= THRESHOLD:
            action = self.choose_action_using_epsilon_greedy_policy(state, epsilon)
            q_table_needs_to_be_updated = True
            reasoning_note = (f"The LLM indicates getting_stuck_likelihood={getting_stuck_likelihood}, which is >= the threshold (80%). "
                    f"The action '{action}' (moving '{self.env.action_name_mapping[action]}') is chosen based on the epsilon_greedy_policy with the epsilon value {epsilon}.")
        else:
            action = self.choose_action_using_greedy_policy(state)
            reasoning_note = (f"The decision to choose the action {action} (moving {self.env.action_name_mapping[action]}) "
                    f"for this step, is based on the greedy policy. It is chosen because its corresponding Q-value, {q_state[action]}, "
                    f"is the highest value in the Q-state {q_state.tolist()} which is associated with the state, i.e. cell, {state}.")
        
        print(f"\n*Note: {reasoning_note}")

        return action, q_table_needs_to_be_updated

    def choose_action_using_epsilon_greedy_policy(self, state, epsilon):
        """ Selects an action using the ϵ-greedy policy to balance exploration and exploitation:
            - With probability ϵ, the agent chooses a random action (exploration).
            - With probability 1-ϵ, the agent chooses the action with the highest Q-value (exploitation).

        This helps the agent avoid getting stuck in local optima early in the learning process.
        """
        if random.uniform(0, 1) < epsilon:
            return random.choice(range(len(self.q_table[state])))  # Exploration: Choose a random action
        else:
            return self.choose_action_using_greedy_policy(state) # Exploitation: Choose the action with the highest Q-value

    def choose_action_using_greedy_policy(self, state):
        """Selects the action with the highest Q-value for a given state."""
        # Using only action = np.argmax(Q[state]) may lead to the situation 
        #  in which consistently selecting the same action when if all items of Q[state] have the same value - 
        #  in this case np.argmax(Q[state]) will always return the index of the first occurrence
        # To avoid that situation, introducing random tie-breaking (the process of breaking a tie,
        # or determining a winner when two or more parties have the same score or result)
        q_state = self.q_table[state]
        shuffled_indices = np.random.permutation(len(q_state)) # Shuffle indices 
        shuffled_q_state = [q_state[i] for i in shuffled_indices]
        action = shuffled_indices[np.argmax(shuffled_q_state)]
        return action
    
    def review_the_movement_log(self, current_state: Tuple, q_state: List, movement_log: List[str], llm) -> Dict[str, Union[str, float, Tuple[int, int], Tuple[int, str]]]:
        """Using the LLM to review the movement log/history for a loop (repeating pattern) detection"""
        try:
            prompt_template = PromptTemplate(input_variables=["current_state", 
                                                            "q_state",
                                                            "movement_history"],
                                            template=prompts.PROMPT_TEMPLATE_GETTING_STUCK_DETECTION)
            
            prompt = prompt_template.format(current_state=str(current_state),
                                            q_state=str(q_state),
                                            movement_history=Utils.list_to_multiline_string(movement_log))
            
            response = llm.invoke(prompt)
        
            json_response = json.loads(response)
            reasoning = json_response["reasoning"]
            getting_stuck_likelihood = json_response["getting_stuck_likelihood_percent"]
            current_state_by_llm = tuple(json_response["current_state"])

            return {"reasoning": reasoning, 
                    "getting_stuck_likelihood": getting_stuck_likelihood, 
                    "current_state": current_state_by_llm}
        
        except Exception as error:
            print(f"\nDEBUG - Error: {error}")
            traceback.print_exc()
            if response:
                print(f"\nDEBUG - The original response from LLM: {response}")

            return {"reasoning": "n/a", 
                "getting_stuck_likelihood": -1, 
                "current_state": current_state,
                "error": f"Error: failed to receive a response from the LLM. An internal error occured: {error}"}

    def update_q_table(self, step: Step):
        INITIAL_ALPHA = 0.1 # Learning rate, controlling how much new information overrides the old  
        DECAY_RATE = 0.01 
        GAMMA = 0.9 # Discount factor, determining the importance of future rewards

        # Appllying a decaying learning rate, which is to ensure Q-Learning to converge to the optimal Q-values
        t = step.step_index
        alpha_t = INITIAL_ALPHA / (1 + DECAY_RATE * t)

        q_index = step.state + (step.action,)
        original_q_value = self.q_table[q_index]
        max_future_q_value = np.max(self.q_table[step.next_state])

        # update
        self.q_table[q_index] += alpha_t * (step.reward + GAMMA * max_future_q_value - original_q_value)

        updated_q_value = self.q_table[q_index]
        return updated_q_value 

class RoboticAgent:
    def __init__(self, env: GridWorldEnv, q_engine: QLearningEngine):
        self.env = env
        self.q_engine = q_engine

    def take_step(self, step_index: int, policy_option: Literal[0, 1, 2], epsilon: float,             
                  movement_log: List[str], llm) -> Tuple[Step, bool, Optional[float]]:
        """This method represents a single iteration of interaction between the agent and the environment.
        It includes Policy Execution (making a decision and choosing an action), Action Execution (transition), 
        and, Learning as it uses the feedback (reward) from the env. and updates the Q table"""

        original_state = self.env.state
        
        q_table_should_to_be_updated = False
        if policy_option == 0: 
            action = self.q_engine.choose_action_using_epsilon_greedy_policy(original_state, epsilon)
            q_table_should_to_be_updated = True
        elif policy_option == 1: # the optimal policy (i.e greedy) 
            action = self.q_engine.choose_action_using_greedy_policy(original_state)
        else: # LLM-assited policy (experimental)
            action, q_table_should_to_be_updated = self.q_engine.choose_action_using_llm_epsilon_greedy_policy(step_index, 
                                                    original_state, epsilon, movement_log, llm)
        
        # The agent will receive a feedback as a reward (or punishment in case of a negative value) from the environment.
        # The state of the environment should be changed also.
        # In other words, perform the transition when taking the action with the current state
        next_state, reward, done, _, _ = self.env.step(action)

        # from here, next_state and self.env.state refer to the same state
        step = Step(step_index=step_index, state=original_state, action=action, reward=reward, next_state=next_state)

        # updated_q_value = None means 'unchanged'
        updated_q_value = self.q_engine.update_q_table(step) if q_table_should_to_be_updated else None

        return step, done, updated_q_value

class Utils:
    def __new__(cls, *args, **kwargs):
        raise TypeError(f"{cls.__name__} cannot be instantiated.")
    
    @staticmethod
    def list_to_multiline_string(lst: List) -> str:
            return "\n".join(lst)