import numpy as np
import matplotlib.pyplot as plt


def visualize_nups(
    s : np.ndarray,
    epsilon,
    nups_epsilons,
    save_path = 'nups.png'
):
    m = s.shape[0]

    # Bound for L_\infty-norm ball
    B1_upper = s + epsilon
    B1_lower = s - epsilon

    # Bound for non-uniform uncertainty set
    B2_upper = s + nups_epsilons
    B2_lower = s - nups_epsilons

    # 3. Plotting
    dims = np.arange(1, m + 1)
    plt.figure(figsize=(12, 6))

    # Plot B1 (The standard box uncertainty)
    plt.fill_between(dims, B1_lower, B1_upper, color='royalblue', alpha=0.15, label='$B_1$: Axis-Aligned')
    plt.plot(dims, B1_upper, '--', color='royalblue', alpha=0.3)
    plt.plot(dims, B1_lower, '--', color='royalblue', alpha=0.3)

    # Plot B2 (The Omega-transformed uncertainty)
    plt.fill_between(dims, B2_lower, B2_upper, color='crimson', alpha=0.25, label='$B_2$: $\Omega$-Weighted')
    plt.plot(dims, B2_upper, color='crimson', linewidth=1.5, alpha=0.7)
    plt.plot(dims, B2_lower, color='crimson', linewidth=1.5, alpha=0.7)

    # Plot Nominal State
    plt.plot(dims, s, '-o', color='black', linewidth=2, label='Nominal $s$', markersize=7)

    # Aesthetics
    plt.title(f"Parallel Coordinate Visualization of Uncertainty Sets ($m={m}$)", fontsize=14)
    plt.xlabel("State Dimension Index", fontsize=12)
    plt.ylabel("Value Range", fontsize=12)
    plt.xticks(dims)
    plt.grid(axis='x', linestyle=':', alpha=0.5)
    plt.legend(loc='upper left', frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def test_visualize_nups():
    m = 11
    s = np.ones(m)
    epsilon = 0.05
    nups_epsilons = np.abs(np.random.randn(m)) * 0.05
    visualize_nups(s, epsilon, nups_epsilons)




def visualize_performance_drop(
    save_path = 'plots/performance_drop.png'
):
    # Data
    beta_labels = ['10', '25', '50', '100', '200', '400', '800']
    x_indices = np.arange(len(beta_labels)) # [0, 1, 2, 3, 4, 5, 6]

    methods = {
        'none':   {'mean': [265, 265, 264, 265, 265, 265, 265], 'std': [6, 6, 6, 6, 6, 6, 6]},
        'random': {'mean': [263, 257, 253, 248, 252, 216, 196], 'std': [8, 14, 34, 39, 18, 71, 56]},
        'critic': {'mean': [262, 250, 251, 244, 238, 210, 167], 'std': [10, 33, 31, 39, 41, 52, 64]},
        'action': {'mean': [260, 243, 243, 231, 229, 182, 129], 'std': [16, 38, 50, 61, 46, 80, 72]}
    }

    plt.figure(figsize=(9, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    for (name, data), color, marker in zip(methods.items(), colors, markers):
        mean = np.array(data['mean'])
        std = np.array(data['std'])
        
        # We plot against x_indices instead of the actual beta values
        plt.plot(x_indices, mean, label=name, color=color, marker=marker, 
                linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
        plt.fill_between(x_indices, mean - std, mean + std, color=color, alpha=0.15)

    # --- THE KEY STEPS FOR EVEN SPACING ---
    plt.xticks(x_indices, beta_labels) 
    # --------------------------------------

    plt.xlabel('Uncertainty Parameter beta$', fontsize=12, fontweight='bold')
    plt.ylabel('Discounted Total Return', fontsize=12, fontweight='bold')
    plt.title('Performance Robustness Comparison', fontsize=14, pad=15)

    # Style tweaks for a "paper" look
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(loc='lower left', frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def test_visualize_performance_drop():
    visualize_performance_drop()



if __name__ == '__main__':
    # test_visualize_performance_drop()
    test_visualize_nups()