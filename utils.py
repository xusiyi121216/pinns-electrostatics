import numpy as np
import matplotlib.pyplot as plt
import torch

def analytical_solution(x, y):
    phi = np.sin(np.pi * x) * np.sinh(np.pi * y) / np.sinh(np.pi)
    return phi

def generate_collocation_points(n_points=3000, device='cuda'):
    x = torch.rand(n_points, 1, device=device)
    y = torch.rand(n_points, 1, device=device)
    return x, y

def generate_boundary_points(n_points=200, device='cuda'):
    n = n_points // 4

    x_bot = torch.rand(n, 1, device=device)
    y_bot = torch.zeros(n, 1, device=device)
    phi_bot = torch.zeros(n, 1, device=device)

    x_top = torch.rand(n, 1, device=device)
    y_top = torch.ones(n, 1, device=device)
    phi_top = torch.sin(torch.pi * x_top)

    x_left = torch.zeros(n, 1, device=device)
    y_left = torch.rand(n, 1, device=device)
    phi_left = torch.zeros(n, 1, device=device)

    x_right = torch.ones(n, 1, device=device)
    y_right = torch.rand(n, 1, device=device)
    phi_right = torch.zeros(n, 1, device=device)

    x_bc = torch.cat([x_bot, x_top, x_left, x_right])
    y_bc = torch.cat([y_bot, y_top, y_left, y_right])
    phi_bc = torch.cat([phi_bot, phi_top, phi_left, phi_right])

    return x_bc, y_bc, phi_bc

def generate_observation_points(n_obs=20, device='cuda'):
    np.random.seed(42)
    x_np = np.random.rand(n_obs, 1)
    y_np = np.random.rand(n_obs, 1)
    phi_np = analytical_solution(x_np, y_np)

    x_obs = torch.tensor(x_np, dtype=torch.float32, device=device)
    y_obs = torch.tensor(y_np, dtype=torch.float32, device=device)
    phi_obs = torch.tensor(phi_np, dtype=torch.float32, device=device)

    return x_obs, y_obs, phi_obs

def plot_results(model, device='cuda', title='PINN Solution'):
    model.eval()

    n = 100
    x_np = np.linspace(0, 1, n)
    y_np = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x_np, y_np)

    x_tensor = torch.tensor(X.flatten()[:, None], dtype=torch.float32, device=device)
    y_tensor = torch.tensor(Y.flatten()[:, None], dtype=torch.float32, device=device)

    with torch.no_grad():
        phi_pred = model(x_tensor, y_tensor).cpu().numpy().reshape(n, n)

    phi_exact = analytical_solution(X, Y)
    error = np.abs(phi_pred - phi_exact)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    c1 = axes[0].contourf(X, Y, phi_pred, levels=50, cmap='jet')
    plt.colorbar(c1, ax=axes[0])
    axes[0].set_title('PINN Prediction')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')

    c2 = axes[1].contourf(X, Y, phi_exact, levels=50, cmap='jet')
    plt.colorbar(c2, ax=axes[1])
    axes[1].set_title('Analytical Solution')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')

    c3 = axes[2].contourf(X, Y, error, levels=50, cmap='hot_r')
    plt.colorbar(c3, ax=axes[2])
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('figures/' + title.replace(' ', '_') + '.png', dpi=150)
    plt.show()

    print(f"L2 Relative Error: {np.linalg.norm(error) / np.linalg.norm(phi_exact):.6f}")