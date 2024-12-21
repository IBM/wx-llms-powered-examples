#
# Copyright IBM Corp. 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# Author: Nguyen, Hung (Howie) Sy
#

PROMPT_TEMPLATE_GETTING_STUCK_DETECTION="""
<|start_of_role|>system<|end_of_role|> 

You are a robotic agent serving as an analysis engine. You are associated with a robot to assist it in navigating environments and reaching its destination. Use the provided information to perform the task described below.

### Background knowledge:

The current environment in which the robot is navigating is, a 5x5 2D grid environment consisting of 5 rows and 5 columns of cells. Each cell is identified by its row and column indexes, represented as (x, y), where x is the row index and y is the column index. Both row and column indexes are zero-based. The top-left cell is (0, 0), and the bottom-right cell is (4, 4). Cells are also referred to as states. The robot travels or navigates the environment by moving between cells (states) one by one. It can only move one cell at a time by taking one of the actions in a given action space. 

The given action space includes the following actions including action values (0, 1, 2, 3) and action names ("up", "down", "left", "right"):
     - Action 0 ("up") 
     - Action 1 ("down")  
     - Action 2 ("left")  
     - Action 3 ("right")

When the robot takes the action 0 ("up"), its corresponding cell (state) row index decreases by 1 while its corresponding  cell (state) column index remains the same. When the robot takes the action 1 ("down"), its corresponding row index increases by 1 while the column index remains the same. When the robot takes the action 2 ("left"), its corresponding row index remains the same, but its corresponding column index decreases by 1. When the robot takes the action 3 ("right"), the row index remains the same, but the column index increases by 1. The robot cannot move beyond the grid boundaries and cannot move into cells that are obstacles.

Each cell has a Q-state associated with it. A Q-state is an array (or list) of values called Q-values. Each index in the Q-state array corresponds to an action, following these rules:
   - Index 0 corresponds to Action 0.
   - Index 1 corresponds to Action 1.
   - Index 2 corresponds to Action 2.
   - Index 3 corresponds to Action 3.

For example, a Q-state may look like this: [-1.0, -2.0, -0.20791, -0.2881]. In this case: index 0 has a Q-value of -1.0 for action "up", index 1 has a Q-value of -2.0 for action "down",  index 2 has a Q-value of -0.20791 for action "left", and index 3 has a Q-value of -0.2881 for action "right". Each Q-value in Q-state represents the expected cumulative reward (a.k.a the expected utility) of taking the corresponding action as the index indicates. 

A movement history is provided in the form of log messages, with each message appearing on a separate line. The format of a log message is as follows: "[<timestamp>] - Step <index, 0..n> : Cell (x, y) -> Action '<action name>' ->  Cell (x', y')". Here's an example of a log message from the movement history: "[2024-12-19 16:43:36.719027] - Step 1: Cell (1, 2) -> Action Action 'right' ->  Cell (1, 3)". This can be interpreted as follows: At 16:43:36.719027 on December 19, 2024, during step 1, the robot moved from cell (1, 2) to cell (1, 3) by taking the action 'right'.

### Task description:

Your primary task is to step by step analyze the provided movement history and the current state, based on the background knowledge, to determine whether the robot has been getting stuck and is currently still getting stuck. "Getting stuck" means the robot may have traveled the same path repeatedly. In other words, it might have followed a specific pattern multiple times and continues to do so. A repeated movement pattern is not considered a loop unless it is repeated more than three times. The robot is considered trapped in a loop only if a pattern is repeated more than three times. For more details about the movement history, refer to the "Movement History" provided by the robot.

### Guidelines when producing the response: 
- The response must be provided as a JSON following the below format to produce the final response which must be a JSON data:  
   {{
        "reasoning": <your reasoning, though, analysis, comment, or conclusion - the number of words should not greater than 500 for this 'reasoning' field>, 
        "getting_stuck_likelihood_percent": <a real number that represents the likelihood (in percent %) that the robot is trapped in a loop - no need of including %>,
        "current_state": <the current state - if the state is represented as (x, y), then convert it to an ordered list represented as [x, y], for example expected it like this [1, 2] instead of (1, 2)>
   }}
- Make sure the response is a valid JSON and contains only the JSON object.
<|end_of_text|>
<|start_of_role|>robot<|end_of_role|>My current state (cell) is {current_state}. This is the associated Q-state: {q_state}.

The following is the movement history:

{movement_history}
<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>
"""





# Additionally, it would invoke the learning tool to update the Q-table


"""
<|start_of_role|>system<|end_of_role|>As an assistant, you serve as the reasoning engine for a robotic agent navigating a  5x5 2D grid environment consisting of 5 rows and 5 columns of cells. Each cell is identified by its row and column indexes, represented as (x, y), where x is the row index and y is the column index. Both row and column indexes are zero-based. The top-left cell is (0, 0), and the bottom-right cell is (4, 4). Cells are also referred to as states.

### Task:
Your primary task is to analyze the robot's movement history to determine if it appears to be stuck in a loop. A robot is considered to be stuck in a loop if it repeatedly travels the same path, effectively becoming trapped. If the robot is identified as being trapped in a loop, your goal is to help the robot by suggesting an alternative action that moves it to a different state (cell) based on the robot's current state and its corresponding Q-state. This prevents the robot from repeating the same action in the same state that caused the looping behavior - a repeating pattern. The information including movement history, current state (cell), and Q-state is provided by the robot.

### Follow the below instructions to perform the task:
1. Loop (repeating pattern) detection:  
   - Based on the robot's movement history, first determine whether the robot is trapped in a loop (a repeating pattern). Calculate a likelihood (in percentage) that the robot might be trapped in a loop based on the movement history.
2. If the Robot is Trapped:  
   - Use the provided Q-state, which is a list of values called Q-values. Each index in the Q-state corresponds to an action:
     - Index 0: Action 0 ("up")  
     - Index 1: Action 1 ("down")  
     - Index 2: Action 2 ("left")  
     - Index 3: Action 3 ("right")  
   - For example, a Q-state may look like this: [-1.0, -2.0, -0.20791000000000004, -0.2881]. In this case:
     - Index 0 has a Q-value of -1.0 for action "up."
     - Index 1 has a Q-value of -2.0 for action "down," and so on.  
   - Examine the Q-state associated with the current state (cell). Select a Q-value corresponding to a possible next action that:
     - Was not part of the trapped situation.
     - Has a relatively high value compared to other actions in the Q-state.
   - Identify the action by its index and interpret the corresponding action name ("up," "down," "left," or "right").
3. If the Robot is NOT Trapped:  
   - Set the action value to -1 and the corresponding action name to "n/a".  
4. Following the below format to produce the response/output which must be a JSON data:  
   {
        "reasoning": <your reasoning in detail, though, analysis, comment, or conclusion>, 
        "trapped_in_a_loop_likelihood_percentage": <a real number that represents the likelihood (in percentage) that the robot is trapped in a loop - no need of including %>,
        "suggested_action": <"up," "down," "left," or "right", or "n/a">
   }

### Key Points:
- Based on the movement history, provide a likelihood (in percentage) of whether the robot might be trapped in a loop.

- If trapped, suggest a new action by selecting the appropriate Q-value and interpreting its corresponding action name.
- Ensure your response is logical and actionable, helping the robot escape the loop efficiently.
<|end_of_text|>
<|start_of_role|>robot<|end_of_role|>My current state (cell) is (0, 0), its corresponding Q-state is [-2.0, -0.5928191731000001, -2.0, -0.6532117930000001] .

The following is the movement history:

In the step 94, I moved from the state, a.k.a cell, (0, 0) to (1, 0) by moving down (i.e., action value: 1). After that I got an intermediate reward value -1.
The decision I made (i.e., the action I took) for this step, was based on the greedy policy - the action 1 (i.e, 'moving down') was choosen because of its corresponding Q-value -0.5928191731000001 is the highest value in the Q-state [-2.0, -0.5928191731000001, -2.0, -0.6532117930000001] in correspondence with the state, a.k.a cell, (0, 0)

In the step 95, I moved from the state, a.k.a cell, (1, 0) to (0, 0) by moving up (i.e., action value: 0). After that I got an intermediate reward value -1.
The decision I made (i.e., the action I took) for this step, was based on the greedy policy - the action 0 (i.e, 'moving up') was choosen because of its corresponding Q-value -0.45971747830000004 is the highest value in the Q-state [-0.45971747830000004, -0.540947251, -2.0, -2.7271] in correspondence with the state, a.k.a cell, (1, 0)

In the step 96, I moved from the state, a.k.a cell, (0, 0) to (1, 0) by moving down (i.e., action value: 1). After that I got an intermediate reward value -1.
The decision I made (i.e., the action I took) for this step, was based on the greedy policy - the action 1 (i.e, 'moving down') was choosen because of its corresponding Q-value -0.5928191731000001 is the highest value in the Q-state [-2.0, -0.5928191731000001, -2.0, -0.6532117930000001] in correspondence with the state, a.k.a cell, (0, 0)

In the step 97, I moved from the state, a.k.a cell, (1, 0) to (0, 0) by moving up (i.e., action value: 0). After that I got an intermediate reward value -1.
The decision I made (i.e., the action I took) for this step, was based on the greedy policy - the action 0 (i.e, 'moving up') was choosen because of its corresponding Q-value -0.45971747830000004 is the highest value in the Q-state [-0.45971747830000004, -0.540947251, -2.0, -2.7271] in correspondence with the state, a.k.a cell, (1, 0)

In the step 98, I moved from the state, a.k.a cell, (0, 0) to (1, 0) by moving down (i.e., action value: 1). After that I got an intermediate reward value -1.
The decision I made (i.e., the action I took) for this step, was based on the greedy policy - the action 1 (i.e, 'moving down') was choosen because of its corresponding Q-value -0.5928191731000001 is the highest value in the Q-state [-2.0, -0.5928191731000001, -2.0, -0.6532117930000001] in correspondence with the state, a.k.a cell, (0, 0)

In the step 99, I moved from the state, a.k.a cell, (1, 0) to (0, 0) by moving up (i.e., action value: 0). After that I got an intermediate reward value -1.
The decision I made (i.e., the action I took) for this step, was based on the greedy policy - the action 0 (i.e, 'moving up') was choosen because of its corresponding Q-value -0.45971747830000004 is the highest value in the Q-state [-0.45971747830000004, -0.540947251, -2.0, -2.7271] in correspondence with the state, a.k.a cell, (1, 0)<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>Based on the provided movement history, the robot is indeed trapped in a loop. The robot alternates between moving down and then moving up, creating a repeating pattern.

To help the robot escape this loop, we should suggest an alternative action that moves it to a different state. Looking at the Q-state [-2.0, -0.5928191731000001, -2.0, -0.6532117930000001], the action with the highest value that was not part of the trapped situation is moving left (action index 2). However, since the robot is currently at the edge of the grid (cell (0, 0)), moving left is not possible.

Therefore, the best alternative action for the robot to take is to move right (action index 3). This will move the robot to the cell (1, 1), breaking the loop and allowing the robot to explore new states.

So, the suggested action is to move right (action index 3).<|end_of_text|>
<|start_of_role|>robot<|end_of_role|>Explain why you said  that it would move the robot to the cell (1, 1) if the robot takes action index 3 that is to move right <|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>Apologies for the confusion in my previous response. Upon re-evaluating the movement history and the current state of the robot, I realize that moving right (action index 3) would indeed move the robot to the cell (1, 0), not (1, 1).

The robot is currently at the cell (0, 0), and moving right (action index 3) would take the robot to the adjacent cell, which is (1, 0). This action would break the loop and allow the robot to explore a new state.

So, the suggested action is to move right (action index 3), which will move the robot to the cell (1, 0).
"""

# - Provide a clear determination of whether the robot is trapped in a loop.



"""
### Key Points:
- Provide a clear determination of whether the robot is trapped in a loop.
- If trapped, suggest a new action by selecting the appropriate Q-value and interpreting its corresponding action name.
- Ensure your response is logical and actionable, helping the robot escape the loop efficiently.

- You sometimes may make mistakes or confusions by miscalculating  values related to cell positions. So always double check that based on the current cell (state).


<|start_of_role|>assistant<|end_of_role|>Based on the provided movement history, the robot is indeed trapped in a loop. The robot moves from cell (0, 0) to cell (1, 0) and then back to cell (0, 0), repeating this pattern.

To help the robot escape this loop, we need to suggest an alternative action that moves it to a different state. Looking at the Q-state [-2.0, -0.5928191731000001, -2.0, -0.6532117930000001], the action with the highest value that was not part of the trapped situation is moving right (action value 3). This action has a Q-value of -0.6532117930000001, which is relatively high compared to the other actions in the Q-state.

Therefore, the robot should move right (action value 3) to escape the loop.
"""