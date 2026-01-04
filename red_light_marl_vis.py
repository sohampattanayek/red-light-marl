import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback 
import pygame
import os

x = 0  
y = 0
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x,y)

MAX_TRACK_LENGTH = 100
SPEED_MOVE = 1.0
MAX_STEPS = 200
TARGET_STREAK = 10
NUM_AGENTS = 5

class SingleLaneEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}
    
    # [MARKER 1: THE RULES OF THE GAME]
    # This class defines the world. It tells the AI what "Red Light" is
    # and calculates rewards: +1 for moving on green, -5 for moving on red.
    def __init__(self):
        super(SingleLaneEnv, self).__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.consecutive_wins = 0
        self.reset_state()

    def reset_state(self):
        self.agent_pos = 0.0
        self.current_step = 0
        self.finished = False

    def get_light_color(self, step):
        if step < 30: return 1.0
        if step < 60: return 0.0
        if step < 100: return 1.0
        if step < 130: return 0.0
        return 1.0

    def reset(self, seed=None, options=None):
        self.reset_state()
        return self._get_obs(), {}

    def _get_obs(self):
        light = self.get_light_color(self.current_step)
        pos = self.agent_pos / MAX_TRACK_LENGTH
        return np.array([light, pos], dtype=np.float32)

    def step(self, action):
        light = self.get_light_color(self.current_step)
        self.current_step += 1
        
        reward = 0.0
        terminated = False
        truncated = False
        
        if action == 1: 
            if light == 0.0: 
                reward = -5.0
                self.agent_pos = 0.0
                self.consecutive_wins = 0 
            else: 
                reward = 1.0
                self.agent_pos += SPEED_MOVE
        else: 
            if light == 0.0: 
                reward = 0.5
            else: 
                reward = -0.1

        if self.agent_pos >= MAX_TRACK_LENGTH:
            self.agent_pos = MAX_TRACK_LENGTH
            if not self.finished:
                reward += 20.0
                self.consecutive_wins += 1
                self.finished = True
                terminated = True
        
        if self.current_step >= MAX_STEPS:
            truncated = True
            if not self.finished:
                self.consecutive_wins = 0 
            
        return self._get_obs(), reward, terminated, truncated, {}

# [MARKER 2: THE VISUALIZER HOOK]
# This is a custom callback. It pauses the math every single step
# to draw the game window. This makes training look cool, but runs slow.
class SquadCheckCallback(BaseCallback):
    def __init__(self, target_streak, render_env):
        super().__init__()
        self.target_streak = target_streak
        self.render_env = render_env 

    def _on_step(self) -> bool:
        streaks = self.training_env.get_attr("consecutive_wins")
        positions = self.training_env.get_attr("agent_pos")
        lights = [env.get_light_color(env.current_step) for env in self.training_env.envs]
        
        self.render_env.render_frame(positions, streaks, lights[0])

        if min(streaks) >= self.target_streak:
            print(f"\n\n*** SQUAD GOALS ACHIEVED! All agents reached {self.target_streak} wins! ***")
            return False 
        return True

class Visualizer:
    def __init__(self):
        pygame.init()
        self.width, self.height = 800, 500
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Training Squad (Parallelized)")
        self.font = pygame.font.Font(None, 28)
        self.colors = [(255, 0, 0), (0, 0, 255), (0, 200, 0), (255, 165, 0), (200, 0, 200)]

    def render_frame(self, positions, streaks, light_val):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()

        bg = (255, 220, 220) if light_val == 0.0 else (220, 255, 220)
        self.screen.fill(bg)
        
        pygame.draw.line(self.screen, (0,0,0), (700, 0), (700, 500), 3)
        
        lane_h = self.height // NUM_AGENTS
        for i in range(NUM_AGENTS):
            y_c = int(i * lane_h + lane_h/2)
            pygame.draw.line(self.screen, (150,150,150), (50, y_c), (700, y_c), 1)
            x = 50 + (positions[i] / MAX_TRACK_LENGTH) * 650
            pygame.draw.circle(self.screen, self.colors[i], (int(x), y_c), 15)
            txt = f"Agent {i+1} Streak: {streaks[i]}"
            self.screen.blit(self.font.render(txt, True, (0,0,0)), (20, y_c - 30))

        status = "TRAINING IN PROGRESS..."
        self.screen.blit(self.font.render(status, True, (0,0,0)), (300, 20))
        pygame.display.flip()

    def close(self):
        pygame.quit()

def victory_lap(model):
    print("Starting Victory Lap...")
    viz = Visualizer()
    pygame.display.set_caption("VICTORY LAP - SLOW MOTION")
    
    envs = [SingleLaneEnv() for _ in range(NUM_AGENTS)]
    obs_list = [env.reset()[0] for env in envs]
    finished = [False] * NUM_AGENTS
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        actions, _ = model.predict(obs_list)
        positions = []
        streaks = [TARGET_STREAK] * NUM_AGENTS 
        light_val = envs[0].get_light_color(envs[0].current_step)
        
        all_done = True
        for i, env in enumerate(envs):
            if not finished[i]:
                obs, _, term, trunc, _ = env.step(actions[i])
                obs_list[i] = obs
                if term or trunc: finished[i] = True
                all_done = False
            positions.append(env.agent_pos)

        viz.render_frame(positions, streaks, light_val)
        clock.tick(15) 
        
        if all_done:
            pygame.time.delay(3000)
            running = False
            
    viz.close()

# [MARKER 3: THE MAIN LOOP]
# Here we wire everything together. We use PPO (the brain)
# and attach the visualizer callback to it.
def main():
    viz = Visualizer()
    env = DummyVecEnv([lambda: SingleLaneEnv() for _ in range(NUM_AGENTS)])
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.005)
    
    callback = SquadCheckCallback(target_streak=TARGET_STREAK, render_env=viz)
    model.learn(total_timesteps=100000, callback=callback)
    
    viz.close()
    victory_lap(model)

if __name__ == "__main__":
    main()