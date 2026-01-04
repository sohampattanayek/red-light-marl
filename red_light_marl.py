import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import pygame
import time
import os

NUM_AGENTS = 5
TARGET_STREAK = 10
MAX_TRACK = 100

class FastTrainingEnv(gym.Env):
    # [MARKER A: THE INVISIBLE GYM]
    # This environment has NO graphics code inside it.
    # It runs pure calculations, which allows the CPU to process
    # thousands of steps per second.
    def __init__(self):
        super(FastTrainingEnv, self).__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.streak = 0
        self.reset_state()

    def reset_state(self):
        self.pos = 0.0
        self.step_count = 0
        self.done = False

    def reset(self, seed=None, options=None):
        self.reset_state()
        return self._get_obs(), {}

    def _get_obs(self):
        light = 1.0 if (self.step_count // 20) % 2 == 0 else 0.0
        return np.array([light, self.pos/MAX_TRACK], dtype=np.float32)

    def step(self, action):
        light = 1.0 if (self.step_count // 20) % 2 == 0 else 0.0
        self.step_count += 1
        
        reward = 0.0
        terminated = False
        truncated = False
        
        if action == 1: 
            if light == 0.0: 
                reward = -5.0
                self.pos = 0.0
                self.streak = 0 
            else: 
                reward = 1.0
                self.pos += 1.0 
        else: 
            if light == 0.0: reward = 0.5 
            else: reward = -0.1 
            
        if self.pos >= MAX_TRACK:
            reward += 20.0
            self.streak += 1
            terminated = True
        
        if self.step_count >= 200:
            truncated = True
            if not terminated: self.streak = 0 
            
        return self._get_obs(), reward, terminated, truncated, {}

# [MARKER B: THE CONSOLE BAR]
# Since we don't have a window, we need to know what's happening.
# This prints a simple progress bar to the terminal so we know
# the AI is actually learning.
class ConsoleProgressBar(BaseCallback):
    def __init__(self, target):
        super().__init__()
        self.target = target
        self.last_print = 0
        
    def _on_step(self) -> bool:
        if self.n_calls - self.last_print > 1000:
            streaks = self.training_env.get_attr("streak")
            min_streak = min(streaks)
            
            bar = "█" * min_streak + "-" * (self.target - min_streak)
            print(f"Current Team Progress: [{bar}] ({min_streak}/{self.target} Wins)")
            
            self.last_print = self.n_calls
            
            if min_streak >= self.target:
                print("\n\n>>> ALL AGENTS TRAINED! Launching Visualization... <<<")
                return False 
        return True

# [MARKER C: THE VICTORY LAP]
# The training is done. NOW we launch Pygame to show off the results.
# We set the window position specifically so it pops up where we want it.
def run_victory_lap(model):
    x = 2000 
    y = 100
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x,y)

    pygame.init()
    screen = pygame.display.set_mode((800, 400))
    pygame.display.set_caption("VICTORY LAP - 5 TRAINED AGENTS")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    envs = [FastTrainingEnv() for _ in range(NUM_AGENTS)]
    obs_list = [env.reset()[0] for env in envs]
    finished = [False] * NUM_AGENTS
    colors = [(255,0,0), (0,0,255), (0,255,0), (255,165,0), (128,0,128)]
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
        actions, _ = model.predict(obs_list)
        
        positions = []
        light_status = 0
        for i, env in enumerate(envs):
            if not finished[i]:
                obs, _, term, trunc, _ = env.step(actions[i])
                obs_list[i] = obs
                light_status = obs[0] 
                if term or trunc: finished[i] = True
            positions.append(env.pos)
            
        bg = (255, 200, 200) if light_status == 0 else (200, 255, 200)
        screen.fill(bg)
        pygame.draw.line(screen, (0,0,0), (700, 0), (700, 400), 5)
        
        for i in range(NUM_AGENTS):
            y = 50 + i * 70
            x = 50 + (positions[i] / MAX_TRACK) * 650
            pygame.draw.circle(screen, colors[i], (int(x), y), 20)
            
        status = "STOP" if light_status == 0 else "GO"
        screen.blit(font.render(status, True, (0,0,0)), (350, 20))
        
        pygame.display.flip()
        clock.tick(15) 
        
        if all(finished):
            pygame.time.delay(2000)
            running = False
    
    pygame.quit()

def main():
    print("Training started... (Graphics disabled for speed)")
    
    env = DummyVecEnv([lambda: FastTrainingEnv() for _ in range(NUM_AGENTS)])
    
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.01)
    callback = ConsoleProgressBar(target=TARGET_STREAK)
    
    model.learn(total_timesteps=100000, callback=callback)
    
    run_victory_lap(model)

if __name__ == "__main__":
    main()