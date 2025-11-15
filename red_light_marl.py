import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import pygame
import copy

# --- Constants for Light Phase Control & Timer ---
LIGHT_MIN_STEPS = 30 	# Minimum 1.0 second (30 steps at 30 FPS)
TIMER_MAX_SECONDS = 15
TIMER_MAX_STEPS = TIMER_MAX_SECONDS * 30 	# 450 steps total
NUM_AGENTS = 5 
MIN_WIN_CONDITION = 3 

# --- Reward constants: SYMMETRICAL PENALTIES ---
MOVE_REWARD_SLOW = 1.0
MOVE_REWARD_FAST = 2.0
STILL_GREEN_PENALTY = -MOVE_REWARD_FAST  # -2.0
RED_LIGHT_MOVEMENT_PENALTY = -MOVE_REWARD_FAST # -2.0 (plus position reset)
RED_LIGHT_STOP_REWARD = MOVE_REWARD_FAST * 2.0 # +4.0


# --- CENTRALIZED MARL ENVIRONMENT (CTDE-ready) ---
class CentralizedRedLightEnv(gym.Env):
    """
    A unified environment where the observation and action spaces are 
    concatenated across all agents, suitable for Centralized Training 
    with a single PPO model (Centralized PPO).
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    # Class variable to track agents finished across all trials/resets
    GLOBAL_FINISHED_AGENTS = set()
    
    def __init__(self, num_agents=NUM_AGENTS, render_mode=None):
        super(CentralizedRedLightEnv, self).__init__()
        
        if num_agents < 1:
            raise ValueError(f"num_agents must be at least 1, got {num_agents}")

        self.num_agents = num_agents
        self.max_position = 100
        self.render_mode = render_mode
        
        # 1. CENTRALIZED OBSERVATION SPACE: Fixed size for PPO
        self.observation_space = spaces.Box(
            low=np.array([0.0] * (1 + self.num_agents)), 
            high=np.array([1.0] + [float(self.max_position)] * self.num_agents), 
            shape=(1 + self.num_agents,), 
            dtype=np.float32
        )
        
        # 2. CENTRALIZED ACTION SPACE: Fixed size for PPO
        self.action_space = spaces.MultiDiscrete([3] * self.num_agents)
        
        # Pygame setup
        if self.render_mode == "human":
            self._init_pygame()
        
        # Initialize state
        self.reset()

    def _init_pygame(self):
        try:
            pygame.init()
            self.window_width = 600
            self.window_height = 200
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption(f"Red Light, Green Light MARL ({self.num_agents} Agents) - Centralized PPO")
            self.clock = pygame.time.Clock()
            pygame.font.init()
            self.font = pygame.font.Font(None, 24)
            self.agent_colors = [
                (255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 165, 0), (128, 0, 128)
            ]
        except Exception as e:
            print(f"Warning: Pygame initialization failed: {e}")
            self.render_mode = None 
            
    def _calculate_red_flashes(self):
        """Calculates two random, separated 1-second (30-step) Red flashes."""
        min_duration = LIGHT_MIN_STEPS
        edge_padding = min_duration + 20
        
        valid_range = TIMER_MAX_STEPS - (2 * min_duration) - (2 * edge_padding)
        if valid_range <= 0:
            schedule = [(50, 50 + min_duration), (TIMER_MAX_STEPS - 50 - min_duration, TIMER_MAX_STEPS - 50)]
            return schedule
        
        possible_starts = list(range(edge_padding, TIMER_MAX_STEPS - edge_padding - min_duration))
        
        if len(possible_starts) < 2:
            schedule = [(50, 50 + min_duration), (TIMER_MAX_STEPS - 50 - min_duration, TIMER_MAX_STEPS - 50)]
            return schedule
            
        max_attempts = 1000
        for attempt in range(max_attempts):
            flash_starts = np.random.choice(possible_starts, size=2, replace=False)
            if abs(flash_starts[0] - flash_starts[1]) >= min_duration + 20:
                break
        else:
            flash_starts = [possible_starts[0], possible_starts[-1]]
            
        schedule = sorted([(s, s + min_duration) for s in flash_starts])
        return schedule
            
    def _get_current_light_status(self):
        """Returns the current light status based on current step."""
        current_step_count = TIMER_MAX_STEPS - self.timer_current_steps
        
        for start_step, end_step in self.red_flash_schedule:
            if start_step <= current_step_count < end_step:
                return 0 	# RED
        
        return 1 	# GREEN

    def _get_obs(self):
        """Returns the efficient centralized observation: [Light Status, Pos_0, Pos_1, ...]."""
        return np.concatenate([
            np.array([self.light_status], dtype=np.float32), 
            self.current_positions 
        ])
    
    def reset(self, seed=None, options=None):
        """Resets the environment for a new trial, excluding permanently finished agents."""
        super().reset(seed=seed)
        
        self.current_positions = np.zeros(self.num_agents, dtype=np.float32)
        self.is_finished = np.zeros(self.num_agents, dtype=bool)
        
        # Permanently exclude globally finished agents
        for i in CentralizedRedLightEnv.GLOBAL_FINISHED_AGENTS:
            self.current_positions[i] = self.max_position  # Keep them visually 'finished'
            self.is_finished[i] = True                     # Keep them marked as finished
        
        self.active_agents = self.num_agents - len(CentralizedRedLightEnv.GLOBAL_FINISHED_AGENTS)
        self.successful_agents_this_trial = 0

        self.timer_current_steps = TIMER_MAX_STEPS
        self.red_flash_schedule = self._calculate_red_flashes()
        self.trial_ended_by_win = False
        self.violation_agent_id = -1
        self.violation_flash_timer = 0
        
        self.stagnation_counter = np.zeros(self.num_agents, dtype=int) 
        
        self.light_status = self._get_current_light_status()
        
        observation = self._get_obs()
        info = {}
        return observation, info
    
    def step(self, actions):
        """
        Takes the combined action for all agents simultaneously.
        """
        
        # 1. Global Updates
        if self.timer_current_steps > 0:
            self.timer_current_steps -= 1
        
        old_light = self.light_status
        self.light_status = self._get_current_light_status()
        
        if old_light != self.light_status:
            current_step_count = TIMER_MAX_STEPS - self.timer_current_steps
            status_word = "RED" if self.light_status == 0 else "GREEN"
            print(f"!!! LIGHT FLIPPED to {status_word} at step {current_step_count}.")
            
        # Update violation flash timer
        if self.violation_flash_timer > 0:
            self.violation_flash_timer -= 1
            if self.violation_flash_timer == 0:
                self.violation_agent_id = -1
                
        # 2. Agent Updates
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        
        for i in range(self.num_agents):
            
            # Skip movement/reward calculation for agents already finished (globally or within this trial)
            if self.is_finished[i]:
                rewards[i] = 0.0
                continue
            
            # --- Per-Step Logic ---
            action = actions[i]
            
            # Determine movement and base reward
            movement = 0
            if action == 1: 
                movement = 0.15
                rewards[i] += MOVE_REWARD_SLOW 
            elif action == 2: 
                movement = 0.3
                rewards[i] += MOVE_REWARD_FAST 
            else: # Stop (action == 0)
                rewards[i] += 0.0 
            
            current_pos = self.current_positions[i]
            
            # --- Light Violation/Stop Checks ---
            
            # 2a. Red Light Logic
            if self.light_status == 0:
                if movement > 0:
                    # Red Light Violation (Penalty + Reset)
                    rewards[i] += RED_LIGHT_MOVEMENT_PENALTY 
                    self.current_positions[i] = 0.0 
                    self.violation_agent_id = i
                    self.violation_flash_timer = 15
                    print(f"Agent {i} VIOLATED red light! Reset to start. (Position was {current_pos:.1f})")
                    continue 
                else:
                    # Red Light Stop (Positive Reward)
                    rewards[i] += RED_LIGHT_STOP_REWARD 
            
            # 2b. Green Light Stop Penalty (STAGNATION)
            elif self.light_status == 1 and movement == 0:
                rewards[i] += STILL_GREEN_PENALTY 
            
            # Update position and check for winning
            new_pos = np.clip(current_pos + movement, 0, self.max_position)
            self.current_positions[i] = new_pos 
            
            if new_pos >= self.max_position:
                if not self.is_finished[i]:
                    rewards[i] = 1000.0
                    self.successful_agents_this_trial += 1
                    CentralizedRedLightEnv.GLOBAL_FINISHED_AGENTS.add(i) 
                    print(f"Agent {i} FINISHED! Total global finished: {len(CentralizedRedLightEnv.GLOBAL_FINISHED_AGENTS)}/{self.num_agents}")
                    
                self.is_finished[i] = True
                self.current_positions[i] = self.max_position
                
                # Check if the MIN_WIN_CONDITION is met
                if len(CentralizedRedLightEnv.GLOBAL_FINISHED_AGENTS) >= MIN_WIN_CONDITION:
                    self.trial_ended_by_win = True
            
        # 3. Episode Termination and Output
        timeout_occurred = self.timer_current_steps <= 0
        
        # Keep timeout penalty to ensure winning is the ultimate goal
        if timeout_occurred:
            # Only penalize agents who are active AND not finished this trial
            for i in range(self.num_agents):
                if i not in CentralizedRedLightEnv.GLOBAL_FINISHED_AGENTS and not self.is_finished[i]:
                    rewards[i] += -600.0 
                            
        total_reward = np.sum(rewards)
        
        # The episode ends if the win condition is met OR if the current trial times out
        terminated = self.trial_ended_by_win or timeout_occurred 
        truncated = False 
        
        observation = self._get_obs()
        info = {} 
        
        return observation, total_reward, terminated, truncated, info
    
    # ... (render and close methods remain the same)
    def render(self):
        if self.render_mode != "human":
            return
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit("User closed the Pygame window.")
        
        BACKGROUND_COLOR_RED = (200, 50, 50)
        BACKGROUND_COLOR_GREEN = (50, 200, 50)
        background_color = BACKGROUND_COLOR_RED if self.light_status == 0 else BACKGROUND_COLOR_GREEN
        self.screen.fill(background_color)
        
        pygame.draw.line(self.screen, (255, 255, 0), 
                             (self.window_width - 5, 0), 
                             (self.window_width - 5, self.window_height), 5)
        
        for i in range(self.num_agents):
            runner_x = self.current_positions[i] * (self.window_width / self.max_position)
            runner_y = 30 + (i * (self.window_height - 60) // max(1, self.num_agents - 1)) if self.num_agents > 1 else self.window_height // 2
            runner_x = np.clip(runner_x, 0, self.window_width)
            
            base_color = self.agent_colors[i % len(self.agent_colors)]
            color = base_color
            
            # Flash white if this agent violated
            if i == self.violation_agent_id and self.violation_flash_timer % 4 < 2:
                color = (255, 255, 255)
            
            # Show finished agents as gold
            if self.is_finished[i]:
                color = (255, 215, 0)
            
            pygame.draw.line(self.screen, color, 
                             (int(runner_x), runner_y + 10), 
                             (int(runner_x), runner_y + 30), 3)
            pygame.draw.circle(self.screen, color, (int(runner_x), int(runner_y)), 7)
        
        light_color = (255, 0, 0) if self.light_status == 0 else (0, 255, 0)
        pygame.draw.circle(self.screen, light_color, (20, 20), 15)
        
        text_color = (255, 255, 255) if self.light_status == 0 else (0, 0, 0)
        
        status_word = "RED" if self.light_status == 0 else ("GREEN")
        debug_text = f"LIGHT: {status_word}"
        debug_surface = self.font.render(debug_text, True, text_color)
        self.screen.blit(debug_surface, (50, 5))
        
        # Display GLOBAL finished count
        success_text = f"Global Finished: {len(CentralizedRedLightEnv.GLOBAL_FINISHED_AGENTS)}/{self.num_agents} (Goal: {MIN_WIN_CONDITION})"
        success_surface = self.font.render(success_text, True, text_color)
        self.screen.blit(success_surface, (self.window_width - 200, 10))
        
        time_remaining = max(0, int(self.timer_current_steps / self.metadata["render_fps"]))
        timer_text = f"Time: {time_remaining:02d}s"
        timer_surface = self.font.render(timer_text, True, text_color)
        self.screen.blit(timer_surface, (self.window_width - 100, self.window_height - 30))
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        
    def close(self):
        if self.render_mode == "human" and hasattr(self, 'screen'):
            pygame.quit()


# --- Training Function for Centralized PPO ---
def train_and_run_marl():
    # Re-introducing fixed step counts
    TOTAL_TIMESTEPS = 500000 
    RENDER_TRAINING_FREQ = 100000 
    
    print("="*60)
    print("CENTRALIZED PPO RED LIGHT GREEN LIGHT TRAINING")
    print("="*60)
    print(f"Number of agents: {NUM_AGENTS}")
    print(f"Win condition: {MIN_WIN_CONDITION}/{NUM_AGENTS} agents must finish.")
    print(f"Total training timesteps limit: {TOTAL_TIMESTEPS}")
    print("="*60)
    
    # Reset the global counter before starting a new training run
    CentralizedRedLightEnv.GLOBAL_FINISHED_AGENTS = set()
    
    def make_env():
        # Note: Render mode is initially None for fast training.
        return CentralizedRedLightEnv(num_agents=NUM_AGENTS, render_mode=None)
        
    vec_env = None
    try:
        # 1. Setup Environment and Model
        vec_env = DummyVecEnv([make_env])
        
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1, 
            device="cpu",
            ent_coef=0.05,  
            n_steps=2048,   
            batch_size=256,
            learning_rate=3e-4,
            gamma=0.99
        )
        
        # 2. Training Loop with Periodic Demo and Early Exit
        trained_steps = 0
        while trained_steps < TOTAL_TIMESTEPS:
            # Check for early exit condition
            if len(CentralizedRedLightEnv.GLOBAL_FINISHED_AGENTS) >= MIN_WIN_CONDITION:
                print(f"\n✅ TRAINING STOPPED EARLY: Minimum win condition ({MIN_WIN_CONDITION}/{NUM_AGENTS}) met after {trained_steps} steps.")
                break
                
            steps_to_train = min(RENDER_TRAINING_FREQ, TOTAL_TIMESTEPS - trained_steps)
            
            print(f"\n--- Training for {steps_to_train} steps (Total: {trained_steps}/{TOTAL_TIMESTEPS}) ---")
            
            model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False)
            trained_steps += steps_to_train
            
            print(f"\n{'='*60}")
            # The demo trial will now persist across resets until the MIN_WIN_CONDITION is met
            print(f"STARTING DEMO TRIALS. Current Global Finished: {len(CentralizedRedLightEnv.GLOBAL_FINISHED_AGENTS)}/{NUM_AGENTS}")
            print(f"{'='*60}")
            # The demo loop keeps running new trials until the global goal is met or interrupted
            run_persistent_demo(NUM_AGENTS, model)
            
    except (SystemExit, KeyboardInterrupt):
        print("\n\nTraining interrupted by user. Models will NOT be saved.")
        return
    finally:
        if vec_env is not None:
            vec_env.close()
    
    # 3. Final Save and Demo (Only runs if the training loop exited)
    if trained_steps >= TOTAL_TIMESTEPS or len(CentralizedRedLightEnv.GLOBAL_FINISHED_AGENTS) >= MIN_WIN_CONDITION:
        print("\nModel saving...")
        try:
            model.save(f"red_light_centralized_ppo_model_final.zip")
            print("Model saved as red_light_centralized_ppo_model_final.zip")
        except Exception as e:
            print(f"Warning: Failed to save model: {e}")
            
        print(f"\n{'='*60}")
        print("FINAL DEMO")
        print(f"{'='*60}")
        # Run one final demo trial using the persistent function
        run_persistent_demo(NUM_AGENTS, model)


def run_persistent_demo(num_agents, model):
    """
    Runs demo trials repeatedly until the MIN_WIN_CONDITION is met 
    (i.e., the loop continues across resets).
    """
    # Check if the global condition is already met
    if len(CentralizedRedLightEnv.GLOBAL_FINISHED_AGENTS) >= MIN_WIN_CONDITION:
        print(f"Goal already met ({MIN_WIN_CONDITION}/{num_agents}). Skipping persistent demo.")
        return
        
    env = CentralizedRedLightEnv(num_agents=num_agents, render_mode="human")
    
    try:
        while len(CentralizedRedLightEnv.GLOBAL_FINISHED_AGENTS) < MIN_WIN_CONDITION:
            print(f"\n--- New Demo Trial Started ---")
            obs, _ = env.reset()
            
            terminated = False
            truncated = False
            
            for step in range(TIMER_MAX_STEPS + 1): 
                
                action, _ = model.predict(obs, deterministic=True)
                
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                
                if terminated:
                    if env.trial_ended_by_win:
                        # The win condition was met *in this trial*, break the inner loop
                        print(f"✓ DEMO SUCCESS: Minimum win condition ({MIN_WIN_CONDITION}/{num_agents}) achieved!")
                        break 
                    else:
                        # Timeout occurred, reset and try again
                        print(f"✗ DEMO TERMINATED (Timeout). Retrying...")
                        break 
                        
            # Pause for 2 seconds to ensure the user can see the final frame before reset or exit.
            env.render()
            if hasattr(pygame, 'time'):
                start_time = pygame.time.get_ticks()
                while pygame.time.get_ticks() - start_time < 2000:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            env.close()
                            raise SystemExit("Demo window closed.")
                    pygame.time.delay(50) 

    except SystemExit:
        print("\nDemo interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    train_and_run_marl()