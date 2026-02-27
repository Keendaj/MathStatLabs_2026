import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def paint_emp(name, samples, sample_sizes, dist, interval):
    a, b = interval
    fig, axes = plt.subplots(1, len(sample_sizes), figsize=(18, 6))
    
    for i, size in enumerate(sample_sizes):
        ax = axes[i]
        sample = samples[size]

        x_sorted = np.sort(sample)
        
        x_full = np.concatenate(([a], x_sorted, [b]))
        y_full = np.concatenate(([0.0], np.arange(1, len(sample)+1)/len(sample), [1.0]))
        
        ax.step(x_full, y_full, where='post', 
                label='Эмпирическая', 
                color='blue', linewidth=1.5)

        if name == "Пуассона":
            grid = np.arange(a, b + 1)
            ax.step(grid, dist.cdf(grid), where='post', 
                    label='Теоретическая', color='red', linewidth=1.8)
        else:
            grid = np.linspace(a, b, 1001)
            ax.plot(grid, dist.cdf(grid), 
                    label='Теоретическая', color='red', linewidth=1.8)
        
        ax.set_xlim(a, b)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f'n = {size}', fontsize=14)
        ax.grid(True, alpha=0.3, ls='--')
        ax.legend(fontsize=10)
    
    fig.suptitle(f'{name} распределение: Функции распределения', fontsize=18, fontweight='bold', y=1.0)
    plt.subplots_adjust(top=0.8)
    plt.tight_layout()
    plt.show()

def paint_core(name, samples, sample_sizes, dist, interval):
    a, b = interval
    fig, axes = plt.subplots(1, len(sample_sizes), figsize=(18, 6))
    
    for i, size in enumerate(sample_sizes):
        ax = axes[i]
        sample = samples[size]

        kde = stats.gaussian_kde(sample)
        if name == "Пуассона":
            grid = np.arange(a, b + 1)
            fine_grid = np.linspace(a, b, 1000)
            ax.plot(fine_grid, kde(fine_grid), label='Ядерная оценка', color='blue')
            ax.plot(grid, dist.pmf(grid), 'o-', label='Теоретическая', color='red')
        else:
            grid = np.linspace(a, b, 1000)
            ax.plot(grid, kde(grid), label='Ядерная оценка', color='blue')
            ax.plot(grid, dist.pdf(grid), label='Теоретическая', color='red')
        
        ax.set_xlim(a, b)
        ax.set_title(f'Размер выборки: {size}', fontsize=14, fontweight='bold')
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('f(x)', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend()
    
    fig.suptitle(f'{name} распределение: Оценки плотности', fontsize=18, fontweight='bold', y=1.0)
    plt.subplots_adjust(top=0.8)
    plt.tight_layout()
    plt.show()

def main():
    random = np.random.default_rng(4)
    
    dist_names = ["Нормальное", "Коши", "Лапласа", "Пуассона", "Равномерное"]
    sample_sizes = [20, 60, 100]
    
    dists = {
        "Нормальное": stats.norm(0, 1),
        "Коши": stats.cauchy(),
        "Лапласа": stats.laplace(0, 1 / np.sqrt(2)),
        "Пуассона": stats.poisson(10),
        "Равномерное": stats.uniform(-np.sqrt(3), 2 * np.sqrt(3))
    }
    
    intervals = {
        "Пуассона": (6, 14)
    }
    for name in dist_names:
        if name not in intervals:
            intervals[name] = (-4, 4)
    
    data = {name: {size: None for size in sample_sizes} for name in dist_names}
    
    for size in sample_sizes:
        data["Нормальное"][size] = random.normal(0, 1, size)
        data["Коши"][size] = random.standard_cauchy(size)
        data["Лапласа"][size] = random.laplace(0, 1 / np.sqrt(2), size)
        data["Пуассона"][size] = random.poisson(10, size)
        data["Равномерное"][size] = random.uniform(-np.sqrt(3), np.sqrt(3), size)
    
    for name in dist_names:
        paint_emp(name, data[name], sample_sizes, dists[name], intervals[name])
        paint_core(name, data[name], sample_sizes, dists[name], intervals[name])
    

if __name__ == "__main__":
    main()