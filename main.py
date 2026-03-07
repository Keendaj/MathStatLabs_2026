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
        
        if name == "Коши":
            a_plot = min(a, x_sorted[0])
            b_plot = max(b, x_sorted[-1])
        else:
            a_plot, b_plot = a, b

        x_full = np.concatenate(([-np.inf], x_sorted, [np.inf]))
        y_full = np.concatenate(([0.0], np.arange(1, len(sample)+1)/len(sample), [1.0]))

        ax.step(x_full, y_full, where='post', 
                label='Эмпирическая', 
                color='blue', linewidth=1.5)
        
        counts, bin_edges = np.histogram(sample, bins='auto')
        cdf_hist = np.cumsum(counts) / len(sample)
        
        bin_edges_ext = np.concatenate(([a_plot], bin_edges, [b_plot]))
        y_hist_ext = np.concatenate(([0.0, 0.0], cdf_hist, [1.0]))
        
        ax.step(bin_edges_ext, y_hist_ext, where='post', 
                label='Теоретическая (гистограмма)', 
                color='green', linestyle='--', linewidth=1.8)

        if name == "Пуассона":
            grid = np.arange(a_plot, b_plot + 1)
            ax.step(grid, dist.cdf(grid), where='post', 
                    label='Теоретическая', color='red', linewidth=1.8)
        else:
            grid = np.linspace(a_plot, b_plot, 1001)
            ax.plot(grid, dist.cdf(grid), 
                    label='Теоретическая', color='red', linewidth=1.8)

        ax.set_xlim(a_plot, b_plot)
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

        if name == "Коши":
            def robust_bw(obj):
                q75, q25 = np.percentile(obj.dataset[0], [75, 25])
                iqr = q75 - q25
                if iqr == 0: iqr = 1.0
                bw = (iqr / 1.34) * (obj.n ** (-1/5)) 
                std = np.std(obj.dataset[0], ddof=1)
                if std == 0: std = 1.0
                return bw / std 
            
            kde = stats.gaussian_kde(sample, bw_method=robust_bw)
        else:
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

def paint_kde_bandwidths(name, samples, sample_sizes, dist, interval, bw_values=[0.2, 0.5, 'scott', 1.5]):
    a, b = interval
    fig, axes = plt.subplots(1, len(sample_sizes), figsize=(18, 6))

    colors = ['red', 'green', 'blue', 'orange', 'purple']
    
    for i, size in enumerate(sample_sizes):
        ax = axes[i]
        sample = samples[size]
        
        if name == "Пуассона":
            grid = np.arange(a, b + 1)
            fine_grid = np.linspace(a, b, 1000)
            ax.plot(grid, dist.pmf(grid), 'o--', label='Теоретическая', color='black', alpha=0.7)
            plot_grid = fine_grid
        else:
            grid = np.linspace(a, b, 1000)
            ax.plot(grid, dist.pdf(grid), label='Теоретическая', color='black', linewidth=2, linestyle='--')
            plot_grid = grid

        for j, bw in enumerate(bw_values):
            kde = stats.gaussian_kde(sample, bw_method=bw)
            
            if isinstance(bw, str):
                label_text = f'KDE (h = {bw})'
            else:
                label_text = f'KDE (h-множ. = {bw})'
                
            ax.plot(plot_grid, kde(plot_grid), label=label_text, 
                    color=colors[j % len(colors)], linewidth=2, alpha=0.8)
        
        ax.set_xlim(a, b)
        ax.set_title(f'n = {size}', fontsize=14, fontweight='bold')
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('f(x)', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(fontsize=10)
    
    fig.suptitle(f'{name} распределение: Влияние ширины окна на ядерную функцию плотности', fontsize=18, fontweight='bold', y=1.0)
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
    
    chosen_dists = ["Пуассона", "Лапласа"]
    
    for name in chosen_dists:
        paint_kde_bandwidths(name, data[name], sample_sizes, dists[name], intervals[name])


if __name__ == "__main__":
    main()