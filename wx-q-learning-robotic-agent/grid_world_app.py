#
# Copyright IBM Corp. 2024-2025
# SPDX-License-Identifier: Apache-2.0
#
# Author: Nguyen, Hung (Howie) Sy
#

import pygame
from typing import List
import textwrap
from datetime import datetime
import constants as Constants
from grid_world_q_learning import Step, Trajectory, GridWorldEnv, QLearningEngine, RoboticAgent

from dotenv import load_dotenv
load_dotenv()

# Loading watsonx libraries
print("Loading watsonx client... Please wait...")
import sys
sys.path.append("../common_libs") # not a good pratice but it's ok in this case
from watsonx import WatsonxClient # type: ignore
print(f"Watsonx URL: {WatsonxClient._get_watsonx_url()}")

class GridWorldUI:
    class RadioButton:
        def __init__(self, x, y, size, label, font, unchecked_color=pygame.Color(200, 200, 200), checked_color=pygame.Color(0, 134, 179), label_color=Constants.WHITE):
            self.circle_center = (x + size // 2, y + size // 2)
            self.radius = size // 2
            self.label = label
            self.font = font
            self.unchecked_color = unchecked_color
            self.checked_color = checked_color
            self.label_color = label_color
            self.selected = False
            self.rect = pygame.Rect(self.circle_center[0] - self.radius, self.circle_center[1] - self.radius, self.radius * 2, self.radius * 2)

        def select(self):
            """Select this radio button."""
            self.selected = True

        def deselect(self):
            """Deselect this radio button."""
            self.selected = False
        
        def get_rect(self):
            return self.rect

        def draw(self, surface):
            """Draw the radio button and label."""
            # Draw the outer circle
            pygame.draw.circle(surface, self.unchecked_color, self.circle_center, self.radius, 2)
            # Draw the inner circle if selected
            if self.selected:
                pygame.draw.circle(surface, self.checked_color, self.circle_center, self.radius - 4)
            # Draw the label
            label_surface = self.font.render(self.label, True, self.label_color)
            surface.blit(label_surface, (self.circle_center[0] + self.radius + 10, self.circle_center[1] - self.radius))

        def handle_event(self, event, radio_buttons):
            """Handle mouse click and update selection state."""
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left mouse click
                # Check if the click is within this button's circle
                if self.rect.collidepoint(event.pos):
                    # Deselect all other radio buttons
                    for button in radio_buttons:
                        button.deselect()
                    # Select this radio button
                    self.select()
    
    class PolicySelector:
        def __init__(self, screen, x, y):
            self.screen = screen
            font = pygame.font.Font(None, 30)
            self.x = x
            self.y = y
            self.height = 40
            self.radio_size = 20
            self.radio_buttons = [
                GridWorldUI.RadioButton(x, y, self.radio_size, "Learning", font),
                GridWorldUI.RadioButton(x, y + self.height, self.radio_size, "The optimal policy", font),
                GridWorldUI.RadioButton(x, y + self.height*2, self.radio_size, "LLM-assisted policy", font),
            ]

        def get_radio_rect(self, index):
            return self.radio_buttons[index].get_rect()

        def render(self, selected_index=0, event=None):
            for i, button in enumerate(self.radio_buttons):
                if i == selected_index:
                    button.selected = True
                else:
                    button.selected = False
                if event:
                    button.handle_event(event, self.radio_buttons)
            for button in self.radio_buttons:
                button.draw(self.screen)

        def get_selected_option(self):
            for i, button in enumerate(self.radio_buttons):
                if button.selected:
                    return i, button.label

    class TextInputBox:
        def __init__(self, x, y, w, h, text=""):
            self.rect = pygame.Rect(x, y, w, h)
            self.font = pygame.font.Font(None, 30)
            self.inactive_color = (100, 100, 100)
            self.active_color = (0, 0, 0)
            self.text_color = Constants.WHITE
            self.color = self.inactive_color
            self.active = False 
            self.text = text

        def handle_event(self, event):
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Toggle active state if clicked inside the input box
                if self.rect.collidepoint(event.pos):
                    self.active = not self.active
                else:
                    self.active = False
                self.color = self.active_color if self.active else self.inactive_color

            if event.type == pygame.KEYDOWN and self.active:
                if event.key == pygame.K_RETURN:
                    pass
                elif event.key == pygame.K_BACKSPACE: 
                    self.text = self.text[:-1]
                else:
                    # Add the typed character to the text
                    self.text += event.unicode

            return self.text

        def draw(self, surface):
            # Render the input box
            pygame.draw.rect(surface, self.color, self.rect, border_radius=5)
            
            # Render the text
            txt_surface = self.font.render(self.text, True, self.text_color)
            
            # Adjust the width of the input box if text is too long
            self.rect.w = max(self.rect.w, txt_surface.get_width() + 10)
            
            # Draw the text
            surface.blit(txt_surface, (self.rect.x + 5, self.rect.y + 5))

    def render_episode_text_input(self):
        self.episode_input_box.draw(self.screen)

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((Constants.WINDOW_WIDTH, Constants.WINDOW_HEIGHT))
        pygame.display.set_caption("Q-Learning Demonstration")
        self.clock = pygame.time.Clock()

        self.screen.fill(Constants.DEFAULT_BACKGROUND_COLOR)

        self.agent_image = pygame.image.load(Constants.AGENT_IMAGE_PATH)
        self.agent_image = pygame.transform.scale(self.agent_image, (Constants.CELL_SIZE - 40, Constants.CELL_SIZE - 40))
        self.goal_image = pygame.image.load(Constants.GOAL_IMAGE_PATH)
        self.goal_image = pygame.transform.scale(self.goal_image, (Constants.CELL_SIZE - 30, Constants.CELL_SIZE - 30))

        self.episode_input_box = GridWorldUI.TextInputBox(Constants.WINDOW_WIDTH - 60 - 20, 10, 60, 34, 
                                                          text=str(Constants.DEFAULT_MAX_EPISODE_NUM))
        self.task_selector = self.PolicySelector(self.screen, x=300, y=550)

    def render_ui(self, state, action, step_index, travel_path, obstacle_hits, 
                  wall_hits, reward, q_value, updated_q_value, episode, episode_number,
                  policy_selected_index, trajectories, scores, is_running=False, is_paused=False):
        self.screen.fill(Constants.DEFAULT_BACKGROUND_COLOR)
        self.render_grid_env(state=state, travel_path=travel_path, 
                        obstacle_hits=obstacle_hits, wall_hits=wall_hits)
        self.render_statistics(state=state, action=action, reward=reward, 
                    wall_hits=wall_hits, episode=episode, episode_number=episode_number,
                    step_index=step_index, obstacle_hits=obstacle_hits, 
                    q_value=q_value, updated_q_value=updated_q_value, scores=scores)
        run_button_rect = self.render_run_button(is_running=is_running)
        pause_button_rect = self.render_pause_button(is_paused) if is_running else None
        self.task_selector.render(selected_index=policy_selected_index)
        self.render_episode_text_input()

        if trajectories and len(trajectories) > 0 and not is_running:
            self.render_finished_text()
        
        return {"run_button_rect": run_button_rect, 
                "pause_button_rect": pause_button_rect}

    def render_grid_env(self, state, travel_path, obstacle_hits, wall_hits):
        """Draw grid area, grid lines, goal, agent and obstacles"""

        # Draw walls surrounding the grid
        wall_rects = [
            # Top wall: Across the entire width of the grid, above the grid
            pygame.Rect(Constants.WALL_THICKNESS, 0, Constants.GRID_SIZE[1] * Constants.CELL_SIZE, Constants.WALL_THICKNESS),
            # Bottom wall: Across the entire width of the grid, below the grid
            pygame.Rect(Constants.WALL_THICKNESS, Constants.WALL_THICKNESS + Constants.GRID_SIZE[0] * Constants.CELL_SIZE, Constants.GRID_SIZE[1] * Constants.CELL_SIZE, Constants.WALL_THICKNESS),
            # Left wall: Along the height of the grid, left of the grid
            pygame.Rect(0, Constants.WALL_THICKNESS, Constants.WALL_THICKNESS, Constants.GRID_SIZE[0] * Constants.CELL_SIZE),
            # Right wall: Along the height of the grid, right of the grid
            pygame.Rect(Constants.WALL_THICKNESS + Constants.GRID_SIZE[1] * Constants.CELL_SIZE, Constants.WALL_THICKNESS, Constants.WALL_THICKNESS, Constants.GRID_SIZE[0] * Constants.CELL_SIZE),
        ]

        if wall_hits is None or not isinstance(wall_hits, (int)):
            wall_hits = 0

        for rect in wall_rects:
            damaged_wall_color = self.adjust_saturation(pygame.Color(Constants.WALL_COLOR), wall_hits * 20 / 100)
            pygame.draw.rect(self.screen, damaged_wall_color, rect)

        # Draw grid area
        grid_area_rect = pygame.Rect(Constants.WALL_THICKNESS, Constants.WALL_THICKNESS, Constants.GRID_SIZE[1] * Constants.CELL_SIZE, Constants.GRID_SIZE[0] * Constants.CELL_SIZE)
        pygame.draw.rect(self.screen, Constants.BLACK, grid_area_rect)

        # Draw grid lines (cells)
        for x in range(Constants.WALL_THICKNESS, Constants.GRID_SIZE[1] * Constants.CELL_SIZE + Constants.WALL_THICKNESS, Constants.CELL_SIZE):
            for y in range(Constants.WALL_THICKNESS, Constants.GRID_SIZE[0] * Constants.CELL_SIZE + Constants.WALL_THICKNESS, Constants.CELL_SIZE):
                rect = pygame.Rect(x, y, Constants.CELL_SIZE, Constants.CELL_SIZE)
                pygame.draw.rect(self.screen, Constants.GRAY, rect, 1)

        # Draw the target
        goal_x = Constants.GOAL[1] * Constants.CELL_SIZE + Constants.WALL_THICKNESS + 15
        goal_y = Constants.GOAL[0] * Constants.CELL_SIZE + Constants.WALL_THICKNESS + 15
        self.screen.blit(self.goal_image, (goal_x, goal_y))

        # Draw obstacles
        if obstacle_hits is None:
            obstacle_hits = {}
        obstacle_hits = sum(obstacle_hits.values())
        for obs in Constants.OBSTACLES:
            obs_cell_color = self.adjust_saturation(pygame.Color(79, 79, 79), obstacle_hits * 10 / 100)
            obs_x = obs[1] * Constants.CELL_SIZE + Constants.WALL_THICKNESS + 10
            obs_y = obs[0] * Constants.CELL_SIZE + Constants.WALL_THICKNESS + 10
            obs_rect = pygame.Rect(obs_x, obs_y, Constants.CELL_SIZE - 20, Constants.CELL_SIZE - 20)
            pygame.draw.rect(self.screen, obs_cell_color, obs_rect)
           
            font = pygame.font.SysFont(None, 36)
            text_surface = font.render(str(obstacle_hits), True, Constants.WHITE)
            text_rect = text_surface.get_rect(center=(obs[1] * Constants.CELL_SIZE + Constants.CELL_SIZE // 2 + Constants.WALL_THICKNESS,
                                                    obs[0] * Constants.CELL_SIZE + Constants.CELL_SIZE // 2 + Constants.WALL_THICKNESS))
            self.screen.blit(text_surface, text_rect) # Display visit count

        # Draw travel_path
        travel_path = travel_path or []
        for step in travel_path:
            if step in Constants.OBSTACLES: continue
            step_x = step[1] * Constants.CELL_SIZE + Constants.WALL_THICKNESS + Constants.CELL_SIZE // 2
            step_y = step[0] * Constants.CELL_SIZE + Constants.WALL_THICKNESS + Constants.CELL_SIZE // 2
            pygame.draw.circle(self.screen, pygame.Color(26, 40, 40), (step_x, step_y), Constants.CELL_SIZE // 6)

        start_x = Constants.WALL_THICKNESS + Constants.CELL_SIZE // 2
        start_y = Constants.WALL_THICKNESS + Constants.CELL_SIZE // 2
        pygame.draw.circle(self.screen, pygame.Color(0, 0, 0) , (start_x, start_y), Constants.CELL_SIZE // 10)

        # Draw agent
        agent_x = state[1] * Constants.CELL_SIZE + Constants.WALL_THICKNESS + 20
        agent_y = state[0] * Constants.CELL_SIZE + Constants.WALL_THICKNESS + 20
        self.screen.blit(self.agent_image, (agent_x, agent_y))

    def render_statistics(self, state=None, action=None, reward=None, 
                          q_value=None, updated_q_value=None, 
                          wall_hits=None, obstacle_hits=None,
                          episode=None, episode_number=Constants.DEFAULT_MAX_EPISODE_NUM,
                          step_index=None, scores=None , note=None):
        """Display statistics and debugging information and scores on the right"""
        font = pygame.font.SysFont(None, 30)
        text_color = Constants.WHITE

        # Adjusted position for the panel (now starts after the right wall)
        statistics_rect_x = Constants.WALL_THICKNESS + Constants.GRID_SIZE[1] * Constants.CELL_SIZE + Constants.WALL_THICKNESS
        statistics_rect = pygame.Rect(statistics_rect_x, 0, Constants.STATISTICS_PANEL_WIDTH, Constants.STATISTICS_PANEL_HEIGHT)
        pygame.draw.rect(self.screen, Constants.DEFAULT_BACKGROUND_COLOR, statistics_rect)

        statistics_lines = []

        if episode is None:
            episode = 0
        statistics_lines.append(f"Episode: {episode + 1}/{episode_number}")
           
        if step_index is not None:
            statistics_lines.append(f"Step: {step_index + 1}")
        if state is not None:
            statistics_lines.append(f"State: {state}")
        if action is not None:
            statistics_lines.append(f"Action: {action}")
        if wall_hits is not None: 
            statistics_lines.append(f"Wall Hits: {wall_hits}")
        if obstacle_hits is not None:
            statistics_lines.append(f"Obstacle Hits: {sum(obstacle_hits.values())}")
        if reward is not None:
            statistics_lines.append(f"Cumulative reward: {reward}")
        if q_value is not None:
            if isinstance(q_value, (int, float)):
                statistics_lines.append(f"Q-Value (before): {q_value:.3f}")
            else:
                statistics_lines.append(f"Q-Value (before): {q_value}")
        if updated_q_value is not None:
            if isinstance(updated_q_value, (int, float)):
                statistics_lines.append(f"Q-Value (after): {updated_q_value:.3f}")
            else:
                statistics_lines.append(f"Q-Value (after): {updated_q_value}")
        if note is not None:
            statistics_lines.append("")
            # Wrap the note text to fit within the debug panel
            wrapped_note = textwrap.wrap(f"Note/comment: {note}", width=30) 
            statistics_lines.extend(wrapped_note)

        # Render debug text
        for i, line in enumerate(statistics_lines):
            text_surface = font.render(line, True, text_color)
            self.screen.blit(text_surface, (statistics_rect_x + 10, 10 + i * 30))

        if scores is not None and isinstance(scores, List):
            # Render scores in a fixed-height area
            score_area_y = 10 + len(statistics_lines) * 30  
            score_area_height = 20  #  2 lines
            pygame.draw.rect(self.screen, Constants.GRAY, (statistics_rect_x, score_area_y, Constants.STATISTICS_PANEL_WIDTH, score_area_height))

            # Add scores text in one line
            scores_text = "Scores (last 3 eps.): " + ", ".join(f"{score}" for score in scores[-3:])
            
            # Render the scores text
            text_surface = font.render(scores_text, True, text_color)
            self.screen.blit(text_surface, (statistics_rect_x + 10, score_area_y + 10))

            # Render line graph below the score text area
            graph_y = score_area_y + score_area_height + 15  
            graph_rect = pygame.Rect(
                statistics_rect_x + 10,
                graph_y, 
                Constants.STATISTICS_PANEL_WIDTH - 20,
                200, 
            )
            self.render_score_graph(scores, graph_rect)
        else:
            scores_text = "Scores (Last 3): n/a"

    def render_score_graph(self, scores, graph_rect):
        """Render the scores as a line graph"""
        # Background for the graph
        pygame.draw.rect(self.screen, (30, 30, 30), graph_rect)

        # Graph dimensions
        padding = 10
        graph_width = graph_rect.width - 2 * padding
        graph_height = graph_rect.height - 2 * padding

        # Draw axes
        pygame.draw.line(self.screen, Constants.WHITE, (graph_rect.x + padding, graph_rect.y + graph_rect.height - padding),
                        (graph_rect.x + graph_rect.width - padding, graph_rect.y + graph_rect.height - padding))  # X-axis
        pygame.draw.line(self.screen, Constants.WHITE, (graph_rect.x + padding, graph_rect.y + padding),
                        (graph_rect.x + padding, graph_rect.y + graph_rect.height - padding))  # Y-axis

        # If no scores, return immediately
        if not scores:
            return

        # Define max_score and min_score
        max_score = max(scores)
        min_score = min(scores)

        # Prevent division by zero if all scores are the same
        if max_score == min_score:
            normalized_scores = [0.5] * len(scores)  # Set all normalized scores to 0.5 if they're equal
        else:
            # Normalize scores to the range [0, 1]
            normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
        
        def scale_point(x, y):
            scaled_x = padding + x / (len(normalized_scores) - 1) * graph_width if len(normalized_scores) > 1 else padding
            scaled_y = graph_rect.y + graph_rect.height - padding - y * graph_height 
            return graph_rect.x + scaled_x, scaled_y
        
        # Plot individual points
        if len(normalized_scores) == 1:
            # Plot the first score if it's the only point
            first_point = scale_point(0, normalized_scores[0])
            pygame.draw.circle(self.screen, pygame.Color(255, 170, 0), (int(first_point[0]), int(first_point[1])), 4)
        else:
            # Plot the scores as a line graph
            points = [scale_point(i, score) for i, score in enumerate(normalized_scores)]
            pygame.draw.lines(self.screen, pygame.Color(0, 134, 179), False, points, 2)  

            for point in points:
                pygame.draw.circle(self.screen, pygame.Color(255, 170, 0), (int(point[0]), int(point[1])), 4)

    def render_pause_button(self, is_paused):
        """Renders the Pause/Continue/Start Learning button dynamically."""
        button_width = 200
        button_height = 50

        # Position the button at the bottom of the Debugging Information area
        button_x = Constants.GRID_SIZE[1] * Constants.CELL_SIZE + (Constants.STATISTICS_PANEL_WIDTH - button_width) // 2 + Constants.WALL_THICKNESS
        button_y = Constants.WINDOW_HEIGHT - Constants.BUTTON_AREA_HEIGHT + (Constants.BUTTON_AREA_HEIGHT - button_height) // 2

        button_text = "Pause" if not is_paused else "Resume"
        button_color = pygame.Color(31, 107, 132) 

        # Draw the button
        text_color = Constants.WHITE
        font = pygame.font.SysFont(None, 36)
        pygame.draw.rect(self.screen, button_color, (button_x, button_y, button_width, button_height))
        text_surface = font.render(button_text, True, text_color)
        text_rect = text_surface.get_rect(center=(button_x + button_width // 2, button_y + button_height // 2))
        self.screen.blit(text_surface, text_rect)

        return pygame.Rect(button_x, button_y, button_width, button_height)

    def render_run_button(self,is_running=False):
        """Displays the Start Execution button below the grid area."""
        button_width = 200
        button_height = 50
        button_x = 300/2 - button_width/2
        button_y = Constants.WINDOW_HEIGHT - Constants.BUTTON_AREA_HEIGHT + (Constants.BUTTON_AREA_HEIGHT - button_height) // 2

        # Define colors and font
        active_button_color = pygame.Color(31, 107, 132)  
        disabled_button_color = pygame.Color(150, 150, 150) 
        text_color = Constants.WHITE
        font = pygame.font.SysFont(None, 36)

        # Set button color and text based on the execution state
        button_color = disabled_button_color if is_running else active_button_color
        if is_running:
            button_text = "Running..." 
        else:  
            button_text = "Run"

        # Draw the button
        pygame.draw.rect(self.screen, button_color, (button_x, button_y, button_width, button_height))
        text_surface = font.render(button_text, True, text_color)
        text_rect = text_surface.get_rect(center=(button_x + button_width // 2, button_y + button_height // 2))
        self.screen.blit(text_surface, text_rect)

        # Return the button's rect for event handling
        return pygame.Rect(button_x, button_y, button_width, button_height)

    def render_note(self, note, position):
        font = pygame.font.SysFont(None, 30)
        text_note = font.render(note, True, pygame.Color(237, 169, 25))
        self.screen.blit(text_note, position)

    def adjust_saturation(self, color: pygame.Color, factor):
        # Get the HSVA values
        h, s, v, a = color.hsva

        # Adjust saturation
        if factor < 0:
            s = s * (1 + factor)  # Decrease saturation
        else:
            s = s + (100 - s) * factor  # Increase saturation

        # Clamp saturation to [0, 100]
        s = max(0, min(100, s))

        # Set the new HSVA values
        new_color = pygame.Color(0, 0, 0, 0) 
        new_color.hsva = (h, s, v, a)

        return new_color

    def render_finished_text(self):
        font = pygame.font.SysFont(None, 36, False, True)
        light_gray = (211, 211, 211)  # RGB for light gray
        dark_gray = (50, 50, 50)      # Background color

        text_surface = font.render("Finished!", True, light_gray, dark_gray)
        # Calculate text position to center it in the grid area
        text_rect = text_surface.get_rect(center=(
            Constants.WALL_THICKNESS + Constants.GRID_SIZE[1] * Constants.CELL_SIZE // 2,
            Constants.WALL_THICKNESS + Constants.GRID_SIZE[0] * Constants.CELL_SIZE // 2
        ))
        self.screen.blit(text_surface, text_rect)

    def update_display(self):
        pygame.display.flip()
        self.clock.tick(Constants.FPS)

class GridWorldApp:
    """Act as the controller to bring all components together, orchestrating the entire system (i.e integration)"""
    def __init__(self):
        self.env = GridWorldEnv()
        self.q_engine = QLearningEngine(self.env)
        self.agent = RoboticAgent(self.env, self.q_engine)
        
        self.ui = GridWorldUI()

        self.episode_number = Constants.DEFAULT_MAX_EPISODE_NUM

    def handle_pause(self, is_paused, pause_button_rect):
        if pause_button_rect:
            # Handle UI events
            for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            exit()
                        elif event.type == pygame.MOUSEBUTTONDOWN and pause_button_rect.collidepoint(event.pos):
                            is_paused = not is_paused
                            self.ui.render_pause_button(is_paused=is_paused)
                            self.ui.update_display()

            # Pausing until Resume...
            while is_paused:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()
                    if event.type == pygame.MOUSEBUTTONDOWN and pause_button_rect.collidepoint(event.pos):
                        is_paused = False
                        self.ui.render_pause_button(is_paused=is_paused)
                        self.ui.update_display()
        
        return is_paused

    def add_to_movement_log(self, movement_log: List[str], step: Step):
        # the size of movement_log should be limitted due to the limit token can accepted by LLM

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        # E.g, [2024-12-19 16:43:36.719027] - Step 1: Cell (1, 2) -> Action Action 'right' ->  Cell (1, 3)
        log_message = f"[{timestamp}] Step {step.step_index}: Cell {step.state} -> Action '{self.env.action_name_mapping[step.action]}' -> Cell {step.next_state}"
        
        print(f"\nDEBUG - movement log: {log_message}")
        movement_log.append(log_message)

    def run_agent(self, overwritten_episode_number = None, 
                     policy_selected_index=0):
        trajectories: List[Trajectory] = []
        is_running = True
        is_paused = False
        scores = []
        pause_button_rect = None

        episode_number = overwritten_episode_number if overwritten_episode_number else self.episode_number

        if policy_selected_index == 0: # learning
            # reset q_table
            self.q_engine.reset()
        else: # not in learning stage
             # for demo purposes
             episode_number = 1

        # Each episode refers to a trial (a run, i.e starting from an initial state until finished)
        for episode in range(episode_number):
            state, _ = self.env.reset()
            episode_path = []
            total_reward = 0
            trajectory = Trajectory(steps=[])
            movement_log: List[str] = []

            for step_index in range(Constants.MAX_STEPS):  # Prevent infinite loops

                is_paused = self.handle_pause(is_paused, pause_button_rect)

                step, is_done, updated_q_value = self.agent.take_step(step_index=step_index, 
                                                 policy_option=policy_selected_index, epsilon=Constants.EPSILON,
                                                 movement_log=movement_log, llm=wx_llm)
                
                original_q_value = self.q_engine.q_table[step.state + (step.action,)]
                updated_q_value = updated_q_value if updated_q_value else 'unchanged'
                step.previous_q_value = original_q_value
                step.current_q_value = updated_q_value

                # Take note of this step
                trajectory.steps.append(step)

                # Add step to path and update total reward for UI
                episode_path.append(state)
                total_reward += step.reward

                if policy_selected_index == 2: # LLM required
                    self.add_to_movement_log(movement_log, step)

                # Render UI
                ui_rect = self.ui.render_ui(state=state, action=step.action, step_index=step_index, 
                travel_path=episode_path, obstacle_hits=self.env.obstacle_hits, wall_hits=self.env.wall_hits, 
                reward=total_reward, q_value=original_q_value, updated_q_value=updated_q_value, 
                episode=episode, episode_number=self.episode_number, policy_selected_index=policy_selected_index, 
                trajectories=trajectories, scores=scores, is_running=is_running, is_paused=is_paused)
                pause_button_rect = ui_rect.get("pause_button_rect")
                self.ui.update_display()

                # Move to the next state
                state = step.next_state
                if is_done:
                    break
                pass

            # before move on to the next episode, the note of the current trajectory
            trajectories.append(trajectory)
            scores.append(total_reward)
            pass
        
        return trajectories, scores

    def run(self):
        policy_selected_index = 0
        last_state = (0, 0)
        last_obstacle_hits = None 
        last_travel_path = []
        last_wall_hits = None
        last_action = "n/a"
        last_reward = "n/a"
        last_episode = -1
        last_step_index = 0
        trajectories = None
        last_q_value="n/a" 
        last_q_updated_value="n/a" 
        scores="n/a"

        global wx_llm
        wx_llm = None

        while True:
            ui_rects = self.ui.render_ui(state=last_state, action=last_action, step_index=last_step_index, 
                travel_path=last_travel_path, obstacle_hits=last_obstacle_hits, wall_hits=last_wall_hits, 
                reward=last_reward, q_value=last_q_value, updated_q_value=last_q_updated_value, 
                episode=last_episode, episode_number=self.episode_number,
                policy_selected_index=policy_selected_index, trajectories=trajectories, scores=scores, is_running=False)
            self.ui.update_display()

            policy_selector = self.ui.task_selector
            ui_rect_learning = policy_selector.get_radio_rect(0)
            ui_rect_greedy = policy_selector.get_radio_rect(1)
            ui_rect_llm = policy_selector.get_radio_rect(2)

            run_button_rect = ui_rects.get("run_button_rect")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.MOUSEBUTTONDOWN and run_button_rect.collidepoint(event.pos):
                    self.is_running = True
                    trajectories, scores = self.run_agent(policy_selected_index=policy_selected_index)
                    self.is_running = False

                    if trajectories:
                        # for debugging information
                        last_trajectory_steps = trajectories[-1].steps
                        last_obstacle_hits = self.env.obstacle_hits 
                        last_state = self.env.state
                        last_step = last_trajectory_steps[-1]
                        last_step_index = len(last_trajectory_steps) - 1
                        last_wall_hits = self.env.wall_hits
                        last_action = last_step.action
                        last_reward = last_step.reward
                        last_episode = len(trajectories) - 1
                        last_q_value=last_step.previous_q_value
                        last_q_updated_value=last_step.current_q_value
                        last_travel_path = []
                        for step in last_trajectory_steps:
                            last_travel_path.append(step.state)
                elif (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and (ui_rect_learning.collidepoint(event.pos) or ui_rect_greedy.collidepoint(event.pos) or ui_rect_llm.collidepoint(event.pos))
                ):                
                    policy_selector.render(event=event)
                    self.ui.update_display()
                    policy_selected_index, _ = policy_selector.get_selected_option()
                    if policy_selected_index > 1 and wx_llm is None: # LLM
                        wx_llm = WatsonxClient.request_llm(
                                model_id="ibm/granite-3-8b-instruct", 
                                decoding_method = "greedy", 
                                temperature = 0.7, 
                                max_new_tokens = 512,
                                stop_sequences=["<|end_of_text|>"])

                episode_number_input_value = self.ui.episode_input_box.handle_event(event)
                if event.type == pygame.KEYDOWN and str(episode_number_input_value).isnumeric():
                    self.episode_number = int(episode_number_input_value)
                    self.ui.update_display()


if __name__ == "__main__":
    app = GridWorldApp()
    app.run()