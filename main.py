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
    (a_ols, b_ols), (a_lad, b_lad) = models
    
    print(f"\n{title}")
    print("-" * 80)
    print(f"{'Метод':<6} | {'a':<8} | {'delta_a':<8} | {'delta_a, %':<10} | {'b':<8} | {'delta_b':<8} | {'delta_b, %':<10}")
    print("-" * 80)
    
    da_ols, da_p_ols, db_ols, db_p_ols = calculate_metrics(a_ols, b_ols)
    print(f"МНК    | {a_ols:<8.3f} | {da_ols:<8.3f} | {da_p_ols:<10.3f} | {b_ols:<8.3f} | {db_ols:<8.3f} | {db_p_ols:<10.3f}")
    
    da_lad, da_p_lad, db_lad, db_p_lad = calculate_metrics(a_lad, b_lad)
    print(f"МНМ    | {a_lad:<8.3f} | {da_lad:<8.3f} | {da_p_lad:<10.3f} | {b_lad:<8.3f} | {db_lad:<8.3f} | {db_p_lad:<10.3f}")
    print("-" * 80)

def plot_single_regression(x, y, models, title, is_outlier_plot=False):
    (a_ols, b_ols), (a_lad, b_lad) = models
    
    plt.figure(figsize=(8, 6))
    
    plt.scatter(x, y, color='black', label='Выборка', zorder=5)
    
    if is_outlier_plot:
        plt.scatter([x[0], x[-1]], [y[0], y[-1]], color='red', marker='x', s=100, zorder=6, label='Выбросы')
        
    plt.plot(x, 2 + 2*x, color='green', linestyle='--', label='Модель (истинная)')
    plt.plot(x, a_ols + b_ols*x, color='red', label='МНК')
    plt.plot(x, a_lad + b_lad*x, color='blue', label='МНМ')
    
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

def main():
    x, y_norm = generate_base_data()
    y_out = create_outliers(y_norm)

    norm = fit_models(x, y_norm)
    out = fit_models(x, y_out)

    print_table("ВЫБОРКА БЕЗ ВЫБРОСОВ", norm)
    print_table("ВЫБОРКА С ВЫБРОСАМИ", out)
    plot_single_regression(x, y_norm, norm, 'Регрессия (без выбросов)')
    plot_single_regression(x, y_out, out, 'Регрессия (с выбросами)', is_outlier_plot=True)

    plt.show()

if __name__ == '__main__':
    main()