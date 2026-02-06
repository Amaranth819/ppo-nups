import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from gymnasium.spaces.discrete import Discrete


'''
    Helper functions
'''
def sample_from_prob(data_list, probabilities = None):
    num_elements = len(data_list)
    idx = np.random.choice(np.arange(num_elements), p = probabilities)
    return data_list[idx]


def pos_to_state_idx(pos, height, width):
    return pos[0] * height + pos[1]


def state_idx_to_pos(state_idx, height, width):
    return (state_idx // width, state_idx % width)


def moveto(pos, action, height, width):
    row, col = pos
    if action == 0: # Up
        row = max(row - 1, 0)
    elif action == 1: # Down
        row = min(row + 1, height - 1)
    elif action == 2: # Left
        col = max(col - 1, 0)
    elif action == 3: # Right
        col = min(col + 1, width - 1)
    return (row, col)


'''
    A tabular MDP example using the grid world.
'''
class GridWorld:
    def __init__(
        self, 
        height = 3, 
        width = 3,
        start_pos = (0, 0),
        goal_pos = (2, 2),
        traps_pos = [(1, 1)],
        goal_reward = 10,
        trap_reward = -10,
        step_reward = -1,
        render = False
    ):
        self.height = height
        self.width = width
        self.world = np.zeros((self.height, self.width))

        # Validate the inputs
        for pos in [start_pos] + [goal_pos] + traps_pos:
            assert pos[0] >= 0 and pos[0] < self.height
            assert pos[1] >= 0 and pos[1] < self.width
        overlap_pos = set([start_pos]).intersection(set([goal_pos])).intersection(set(traps_pos))
        assert len(overlap_pos) == 0

        # Attributes      
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.traps_pos = traps_pos

        self.goal_reward = goal_reward
        self.trap_reward = trap_reward
        self.step_reward = step_reward

        self.render = render

        # State and action space
        self.state_space = Discrete(n = self.height * self.width)
        self.action_strings = {0 : 'Up', 1 : 'Down', 2 : 'Left', 3 : 'Right'}
        self.action_space = Discrete(n = len(self.action_strings))
        
        # States
        self.curr_pos = start_pos
        self.ep_reward, self.ep_length = 0, 0

        # map (S: Start, G: Goal, T: Trap, E: Empty)
        self.map = [['E' for _ in range(self.width)] for _ in range(self.height)]
        self.map[self.start_pos[0]][self.start_pos[1]] = 'S'
        self.map[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        for pos in self.traps_pos:
            self.map[pos[0]][pos[1]] = 'T'

        # Render the grid world
        if self.render:
            self.open_render()


    '''
        Render functions
    '''
    def open_render(self):
        self.render = True

        # Setup the figure once
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(self.width, self.height+2))
        self._setup_background()
        
        # Create the agent object once
        self.agent_marker = plt.Circle(self._get_coords(self.curr_pos), 0.3, color='blue', label='Agent')
        self.ax.add_patch(self.agent_marker)

        # Legend elements
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label=f'Agent (Step Reward = {self.step_reward})'),
            mpatches.Patch(color='green', alpha=0.3, label=f'Goal (Reward = {self.goal_reward})'),
            mpatches.Patch(color='red', alpha=0.3, label=f'Trap (Reward = {self.trap_reward})')
        ]
        self.ax.legend(
            handles=legend_elements, 
            loc='upper center', 
            bbox_to_anchor=(0.5, -0.05), 
            ncol=1
        )

        # Text for showing the current status
        self.status_text = self.fig.text(0.5, 0.02, f"Ep_Reward={self.ep_reward} | Ep_Length={self.ep_length}", ha='center', fontsize=10, fontweight='bold', color='darkblue')

        # Tight layout
        plt.tight_layout()


    def close_render(self):
        self.render = False
    

    def _get_coords(self, pos):
        """Converts (row, col) to Matplotlib (x, y) (the corner coordinate)"""
        return (pos[1], self.height - 1 - pos[0])
    

    def _setup_background(self):
        # Draw grid lines and special zones (static)
        for x in range(self.height + 1):
            self.ax.axhline(x, lw=2, color='black')
        for x in range(self.width + 1):
            self.ax.axvline(x, lw=2, color='black')
        
        # Goal and traps
        self.ax.add_patch(plt.Rectangle(self._get_coords(self.goal_pos), 1, 1, color='green', alpha=0.3))
        for trap_pos in self.traps_pos:
            self.ax.add_patch(plt.Rectangle(self._get_coords(trap_pos), 1, 1, color='red', alpha=0.3))
        
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect('equal') # Keeps the grid squares square
        self.ax.axis('off') # Hides the axis numbers for a cleaner look


    def render_func(self):
        # Updates only the agent's center point
        agent_corner_pos = self._get_coords(self.curr_pos)
        agent_circle_center = (agent_corner_pos[0] + 0.5, agent_corner_pos[1] + 0.5)
        self.agent_marker.set_center(agent_circle_center)

        # Update the status text
        self.status_text.set_text(f"Ep_Reward={self.ep_reward} | Ep_Length={self.ep_length}")

        # Draw the canvas (call this at the end)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    '''
        Simulation
    '''
    def reset(self, verbose = False):
        self.curr_pos = self.start_pos
        self.ep_reward, self.ep_length = 0, 0
        if self.render:
            self.render_func()
        if verbose:
            print(f"Starting at: {self.curr_pos}")
        # return pos_to_state_idx(self.curr_pos, self.height, self.width)
        return self.curr_pos
    

    def step(self, action, verbose = False):
        """
        Actions: 0=Up, 1=Down, 2=Left, 3=Right
        """
        last_pos = self.curr_pos
        self.curr_pos = moveto(self.curr_pos, action, self.height, self.width)
        
        # Determine Reward and Termination
        rew, done = self.reward_func(self.curr_pos)

        # Update the episode information
        self.ep_length += 1
        self.ep_reward += rew

        if self.ep_length >= 100:
            done = True

        if done:
            info = {'ep_reward' : self.ep_reward, 'ep_length' : self.ep_length, 'curr_pos' : self.curr_pos} 
        else:
            info = {'curr_pos' : self.curr_pos}

        if verbose:
            print(f"Action: {action} ({self.action_strings[action]}) | From State {last_pos} To State {self.curr_pos} | Reward: {rew} | Done: {done} | Ep_Reward: {self.ep_reward} | Ep_Length: {self.ep_length}")

        # Render the GUI
        if self.render:
            self.render_func()

        # return pos_to_state_idx(self.curr_pos, self.height, self.width), rew, done, info
        return self.curr_pos, rew, done, info
    

    def reward_func(self, pos):
        char = self.map[pos[0]][pos[1]]
        if char == 'G':
            return self.goal_reward, True
        elif char == 'T':
            return self.trap_reward, True
        else:
            return self.step_reward, False

    

def test_gridworld():
    # --- Quick Test ---
    env = GridWorld(
        height = 3, 
        width = 3,
        start_pos = (0, 0),
        goal_pos = (2, 2),
        traps_pos = [(1, 1)],
        goal_reward = 10,
        trap_reward = -100,
        step_reward = -1,
        render = True
    )


    for _ in range(1):
        env.reset(verbose = True)
        time.sleep(1)
        while True: 
            move = env.action_space.sample()
            next_state, reward, done, info = env.step(move, verbose = True)
            time.sleep(1)
            if done:
                break



'''
    A Q-learning agent
'''
class QLearningAgent:
    def __init__(
        self, 
        world_shape, 
        action_space : Discrete, 
        lr=0.1, 
        gamma=0.99, 
        epsilon=0.1,
        seed=42,
    ):
        self.world_height, self.world_width = world_shape
        self.q_table = np.zeros((self.world_height, self.world_width, action_space.n), dtype=np.float32)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon # Exploration rate
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)


    def choose_action(self, state):
        rn = self.rng.uniform(0, 1)
        # Epsilon-greedy policy
        if rn < self.epsilon:
            # Exploration
            return self.action_space.sample()
        else:
            # Exploitation
            return self.policy(state)


    def update(self, s, action, rew, s_next):
        r, c = s
        nr, nc = s_next
        # Q-learning update
        delta = (1-self.gamma) * rew + self.gamma * np.max(self.q_table[nr, nc, :]) - self.q_table[r, c, action]
        self.q_table[r, c, action] += self.lr * delta


    def policy(self, state):
        r, c = state
        max_ids = np.where(self.q_table[r, c, :] == np.max(self.q_table[r, c, :]))[0]
        return self.rng.choice(max_ids)   


    def save(self, save_path = 'QAgent.npz'):
        data_dict = {
            'q_table' : self.q_table,
            'lr' : self.lr,
            'gamma' : self.gamma,
            'epsilon' : self.epsilon,
        }
        np.savez(save_path, **data_dict)
        print(f'Save the agent to {save_path}!')


    def load(self, load_path):
        loaded_data = np.load(load_path)
        self.q_table = loaded_data['q_table']
        self.lr = loaded_data['lr']
        self.gamma = loaded_data['gamma']
        self.epsilon = loaded_data['epsilon']
        print(f'Load the agent from {load_path}!')



def q_learning(
    agent : QLearningAgent, 
    env : GridWorld,
    state_perturbation_func = None,
    episodes = 100
):
    episode_iters = np.arange(episodes)+1
    episode_rewards = []

    env.close_render()
    for ep in episode_iters:
        state = env.reset()
        done = False
        
        while not done:
            # Step the simulation
            if state_perturbation_func is not None:
                state = state_perturbation_func(state)
            action = agent.choose_action(state)
            next_state, rew, done, info = env.step(action)
            
            # Update Q-Table
            agent.update(state, action, rew, next_state)

            # Update the current state
            state = next_state

        episode_rewards.append(info['ep_reward'])

        if ep % 10 == 0:
            print(f'Finished {ep} episodes!')


    plt.plot(episode_iters, episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.title('Training Rewards Curve')
    plt.savefig('training_curve.png')
    plt.close()


'''
    Analysis
'''
def calculate_advantages(q_table):
    v_table = np.max(q_table, axis = -1, keepdims = True)
    adv_table = q_table - v_table
    return adv_table


def calculate_total_discounted_return(step_rewards, gamma):
    coef = gamma ** np.arange(len(step_rewards))
    return np.sum(step_rewards * coef)


def plot_q_values_map(
    policy_qs,
    policy, 
    height, 
    width, 
    goal, 
    traps, 
    normalize_q_values = True,
    save_fig = 'q_values_map.png'
):
    """
    Plots a heatmap of max Q-values with arrows indicating the best policy.
    """
    if normalize_q_values:
        policy_qs /= np.max(policy_qs)

    # Action mapping (matching your GridWorld: 0=Up, 1=Down, 2=Left, 3=Right)
    action_arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 2. Plot heatmap of Q-values
    im = ax.imshow(policy_qs, cmap="YlGn", interpolation='nearest')
    fig.colorbar(im, ax=ax, label='Max Q-Value')

    # 3. Annotate with arrows and special markers
    for r in range(height):
        for c in range(width):
            # Check if state is a goal or trap (where arrows aren't needed)
            if (r, c) == goal:
                ax.text(c, r, '★', ha='center', va='center', color='gold', fontsize=20)
            elif (r, c) in traps:
                ax.text(c, r, '✖', ha='center', va='center', color='red', fontsize=20)
            else:
                # Draw the arrow for the best action
                arrow = action_arrows[policy((r, c))]
                # Also show the numerical value for precision
                ax.text(c, r, f"{arrow}\n{policy_qs[r, c]:.2f}", 
                        ha='center', va='center', color='black', fontsize=10)

    # 4. Formatting
    ax.set_xticks(np.arange(width))
    ax.set_yticks(np.arange(height))
    ax.set_title("Q-Function Policy Map" + " (Normalized)" if normalize_q_values else "")
    plt.savefig(save_fig)
    plt.close()


'''
    Apply state perturbation to a policy
'''
def apply_state_perturbation(pos):
    # 
    r, c = pos
    r = max(r - 1, 0)
    return (r, c)



# class StatePerturbedQLearningAgent(QLearningAgent):
#     def __init__(self, world_shape, action_space, lr=0.1, gamma=0.99, epsilon=0.1, seed=42):
#         super().__init__(world_shape, action_space, lr, gamma, epsilon, seed)


#     def policy(self, state):
#         state = apply_state_perturbation(state)
#         return super().policy(state)


if __name__ == '__main__':
    env = GridWorld(
        height = 4, 
        width = 4,
        start_pos = (0, 0),
        goal_pos = (3, 3),
        traps_pos = [(1, 1), (1, 3), (2, 3), (3, 0)],
        goal_reward = 1,
        trap_reward = 0,
        step_reward = 0,
        render = False
    )

    gamma = 0.95
    agent = QLearningAgent(
        world_shape = (env.height, env.width),
        action_space = env.action_space,
        lr = 0.8,
        gamma = gamma,
        epsilon = 0.1
    )

    
    # Training
    q_learning(agent = agent, env = env, episodes = 2000)
    agent.save(save_path = 'QAgent.npz')

    # Testing 
    agent.load(load_path = 'QAgent.npz')
    # env.open_render()


    clean_policy_episode_rewards = []
    for ep in np.arange(10)+1:
        state = env.reset(verbose = False)
        done = False
        # time.sleep(1)
        # input()

        ep_rewards = []
        while not done:
            # Step the simulation
            # print(agent.q_table[state])
            action = agent.policy(state)
            next_state, rew, done, info = env.step(action, verbose = False)
            state = next_state
            # time.sleep(1)
            ep_rewards.append(rew)
        clean_policy_episode_rewards.append(ep_rewards)

        # print(f'Test episode {ep}: Ep_reward = {info["ep_reward"]} | Ep_length = {info["ep_length"]}')   

    # Create a clean policy from the Q-Learning agent
    # Reshape max Q-values into the grid shape
    clean_policy_maxqs = np.max(agent.q_table, axis=-1)
    plot_q_values_map(clean_policy_maxqs, agent.policy, env.height, env.width, env.goal_pos, env.traps_pos, save_fig = 'q_values_map_clean_policy.png')
    clean_discounted_returns = [calculate_total_discounted_return(rewards, gamma) for rewards in clean_policy_episode_rewards]
    print(f'Discounted returns: {np.mean(clean_policy_episode_rewards)} +- {np.std(clean_policy_episode_rewards)}')


    # Apply state perturbation to the policy
    def perturbed_policy(state):
        perturbed_state = apply_state_perturbation(state)
        return agent.policy(perturbed_state)
    
    perturbed_policy_episode_rewards = []
    perturbed_policy_advs = []

    adv_table = calculate_advantages(agent.q_table)

    for ep in np.arange(10)+1:
        state = env.reset(verbose = False)
        done = False
        # time.sleep(1)
        
        ep_rewards = []
        sp_advs = []

        while not done:
            # Step the simulation
            # print(agent.q_table[state])
            action = perturbed_policy(state)
            next_state, rew, done, info = env.step(action, verbose = False)
            state = next_state
            # time.sleep(1)
            ep_rewards.append(rew)

            sp_advs.append(adv_table[state[0], state[1], action])
        
        perturbed_policy_episode_rewards.append(ep_rewards)
        perturbed_policy_advs.append(np.array(sp_advs))

    plot_q_values_map(clean_policy_maxqs, perturbed_policy, env.height, env.width, env.goal_pos, env.traps_pos, save_fig = 'sp_q_values_map_clean_policy.png')
    sp_discounted_returns = [calculate_total_discounted_return(rewards, gamma) for rewards in perturbed_policy_episode_rewards]
    print(f'Discounted returns: {np.mean(sp_discounted_returns)} +- {np.std(sp_discounted_returns)}')



    est_performance_diff = np.concatenate(perturbed_policy_advs).mean() / (1-gamma)
    print(est_performance_diff)