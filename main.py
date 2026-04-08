import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def mlm_objective(params, x, y):
    a, b = params
    return np.sum(np.abs(y - (a + b * x)))

def calculate_metrics(a_est, b_est, a_true=2.0, b_true=2.0):
    delta_a = np.abs(a_true - a_est)
    delta_a_pct = (delta_a / np.abs(a_true)) * 100
    
    delta_b = np.abs(b_true - b_est)
    delta_b_pct = (delta_b / np.abs(b_true)) * 100
    
    return delta_a, delta_a_pct, delta_b, delta_b_pct

def fit_models(x, y):
    b_lsm, a_lsm = np.polyfit(x, y, 1)
    
    res = opt.minimize(mlm_objective, [a_lsm, b_lsm], args=(x, y))
    a_mlm, b_mlm = res.x
    
    return (a_lsm, b_lsm), (a_mlm, b_mlm)

def generate_base_data(size=20, a_true=2.0, b_true=2.0, seed=4):
    x = np.linspace(-1.8, 2.0, size)
    np.random.seed(seed)
    eps = np.random.normal(0, 1, size=size)
    y = a_true + b_true * x + eps
    return x, y

def create_outliers(y):
    y_out = y.copy()
    y_out[0] += 10
    y_out[-1] -= 10
    return y_out


def print_table(title, models):
    (a_lsm, b_lsm), (a_mlm, b_mlm) = models
    
    print(f"\n{title}")
    print("-" * 80)
    print(f"{'Метод':<6} | {'a':<8} | {'delta_a':<8} | {'delta_a, %':<10} | {'b':<8} | {'delta_b':<8} | {'delta_b, %':<10}")
    print("-" * 80)
    
    da_lsm, da_p_lsm, db_lsm, db_p_lsm = calculate_metrics(a_lsm, b_lsm)
    print(f"МНК    | {a_lsm:<8.3f} | {da_lsm:<8.3f} | {da_p_lsm:<10.3f} | {b_lsm:<8.3f} | {db_lsm:<8.3f} | {db_p_lsm:<10.3f}")
    
    da_mlm, da_p_mlm, db_mlm, db_p_mlm = calculate_metrics(a_mlm, b_mlm)
    print(f"МНМ    | {a_mlm:<8.3f} | {da_mlm:<8.3f} | {da_p_mlm:<10.3f} | {b_mlm:<8.3f} | {db_mlm:<8.3f} | {db_p_mlm:<10.3f}")
    print("-" * 80)

def plot_regression_subplot(ax, x, y, models, title, is_outlier_plot=False):

    (a_lsm, b_lsm), (a_mlm, b_mlm) = models
    
    ax.scatter(x, y, color='black', label='Выборка', zorder=5)
    
    if is_outlier_plot:
        ax.scatter([x[0], x[-1]], [y[0], y[-1]], color='red', marker='x', s=100, zorder=6, label='Выбросы')
        
    ax.plot(x, 2 + 2*x, color='green', linestyle='--', label='Модель (истинная)')
    ax.plot(x, a_lsm + b_lsm*x, color='red', label='МНК')
    ax.plot(x, a_mlm + b_mlm*x, color='blue', label='МНМ')
    
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)

def plot_all_results(x, y_norm, models_norm, y_out, models_out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    plot_regression_subplot(axes[0], x, y_norm, models_norm, 'Регрессия (без выбросов)')
    plot_regression_subplot(axes[1], x, y_out, models_out, 'Регрессия (с выбросами)', is_outlier_plot=True)
    
    plt.tight_layout()
    plt.show()

def main():
    x, y_norm = generate_base_data()
    y_out = create_outliers(y_norm)

    models_norm = fit_models(x, y_norm)
    models_out = fit_models(x, y_out)

    print_table("ВЫБОРКА БЕЗ ВЫБРОСОВ", models_norm)
    print_table("ВЫБОРКА С ВЫБРОСАМИ", models_out)

    plot_all_results(x, y_norm, models_norm, y_out, models_out)

if __name__ == '__main__':
    main()