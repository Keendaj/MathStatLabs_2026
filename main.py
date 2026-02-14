import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def calculate_bins(data):
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    if iqr > 0:
        h_fd = 2 * iqr / (len(data) ** (1/3))
        bins = int((max(data) - min(data)) / h_fd)
    else:
        bins = 10

    bins = min(bins, 100)
    unique_values = len(np.unique(data))
    bins = min(bins, unique_values)
    return max(2, bins)

def get_theoretical(name, x_range):
    if name == "Нормальное":
        return stats.norm.pdf(x_range, 0, 1)
    elif name == "Коши":
        return stats.cauchy.pdf(x_range, 0, 1)
    elif name == "Лапласа":
        return stats.laplace.pdf(x_range, 0, 1/np.sqrt(2))
    elif name == "Пуассона":
        x_disc = np.arange(0, 21)
        return x_disc, stats.poisson.pmf(x_disc, 10)
    elif name == "Равномерное":
        return stats.uniform.pdf(x_range, -np.sqrt(3), 2*np.sqrt(3))

def paint_distribution(name, data_list, sample_sizes):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    hist_color = '#3498db'
    true_color = '#e74c3c'
    
    for idx, (data, size) in enumerate(zip(data_list, sample_sizes)):
        ax = axes[idx]
        
        bins_count = calculate_bins(data)

        if name == "Коши":
            vmin, vmax = np.percentile(data, [1, 99])
            data_clean = data[(data >= vmin) & (data <= vmax)]
            
            x_range = np.linspace(vmin, vmax, 1000)
            
            bins_count = calculate_bins(data_clean)
            ax.hist(data_clean, bins=bins_count, density=True, color=hist_color,
                    alpha=0.7, edgecolor='white', linewidth=1, label='Случайное (обрезано 1-99%)')
        elif name == "Пуассона":
            unique, counts = np.unique(data, return_counts=True)
            ax.bar(unique, counts/len(data), width=0.8, color=hist_color, 
                   alpha=0.7, edgecolor='white', linewidth=1, label='Случайное')
        else:
            ax.hist(data, bins=bins_count, density=True, color=hist_color,
                    alpha=0.7, edgecolor='white', linewidth=1, label='Случайное')
        
        if name == "Пуассона":
            x_theor, y_theor = get_theoretical(name, None)
            ax.scatter(x_theor, y_theor, color=true_color, s=100, zorder=5,
                      label='Теоретическое', marker='o')
            ax.vlines(x_theor, 0, y_theor, colors=true_color, alpha=0.5, linewidth=2)
        else:
            if name == "Коши":
                x_range = np.linspace(-10, 10, 1000)
            elif name == "Равномерное":
                x_range = np.linspace(-2, 2, 1000)
            elif name == "Лапласа":
                x_range = np.linspace(-4, 4, 1000)
            else:
                x_range = np.linspace(min(data), max(data), 1000)
            
            y_theor = get_theoretical(name, x_range)
            ax.plot(x_range, y_theor, color=true_color, linewidth=3, 
                   label='Теоретическое')
        
        ax.set_title(f'Размер выборки: {size}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Значение', fontsize=11)
        ax.set_ylabel('Плотность вероятности', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(fontsize=10)
    
    
    fig.suptitle(f'{name} распределение', fontsize=18, fontweight='bold', y=1.0)
    plt.subplots_adjust(top=0.8)
    plt.tight_layout()
    plt.show()
    
    return

def calculate_characteristics(data):
    mean = np.mean(data)
    median = np.median(data)
    zR = (np.min(data) + np.max(data)) / 2
    q14, q34 = np.percentile(data, [25, 75])
    zQ = (q14 + q34) / 2
    n = len(data)
    trim_count = int(n * 0.1)
    if trim_count > 0:
        sorted_data = np.sort(data)
        trimmed = sorted_data[trim_count: -trim_count]
        ztr = np.mean(trimmed)
    
    return { 'mean': mean, 'median': median, 'zR': zR, 'zQ': zQ, 'ztr': ztr }

def print_results_table(dist_name, results, size):
    print("")
    print(f" {dist_name} РАСПРЕДЕЛЕНИЕ (n = {size})".center(69))
    print(f"| {'Оценка':<25} | {'Среднее':>12} | {'Дисперсия':>12} | {'<x> +- D':>14} |")
    print(f"|{'-'*27}|{'-'*14}|{'-'*14}|{'-'*16}|")
    
    names = [ 'mean', 'median', 'zR', 'zQ', 'ztr' ]
    
    for name in names:
        mean_val = results[name]['mean']
        var_val = results[name]['variance']
        mean_rounded = round(mean_val, 1)
        std_rounded = round(var_val, 1)
        error_str = f"{mean_rounded:.1f} +- {std_rounded:.1f}"
        print(f"| {name:<25} | {mean_val:>12.6f} | {var_val:>12.6f} | {error_str:>14} |")

def main():
    random = np.random.default_rng(3)
    sample_sizes = [10, 100, 1000]
    n_simulations = 1000
    
    distributions = {
        'Нормальное': lambda n: random.normal(0, 1, n),
        'Коши': lambda n: random.standard_cauchy(n),
        'Лапласа': lambda n: random.laplace(0, 1/np.sqrt(2), n),
        'Пуассона': lambda n: random.poisson(10, n),
        'Равномерное': lambda n: random.uniform(-np.sqrt(3), np.sqrt(3), n)
    }

    for dist_name, dist_func in distributions.items():
        for size in sample_sizes:
            all_results = { 'mean': [], 'median': [], 'zR': [], 'zQ': [], 'ztr': [] }
            
            for _ in range(n_simulations):
                sample = dist_func(size)
                stats = calculate_characteristics(sample)
                
                for key in all_results:
                    all_results[key].append(stats[key])
            
            size_results = {}
            for key, values in all_results.items():
                size_results[key] = {
                    'mean': np.mean(values),
                    'variance': np.var(values, ddof=1)
                }

            print_results_table(dist_name, size_results, size)
    
    random = np.random.default_rng(3)

    data = { "Нормальное": [], "Коши": [], "Лапласа": [], "Пуассона": [], "Равномерное": [] }
    sample_sizes = [10, 100, 1000]
    
    for i in sample_sizes:
        data["Нормальное"].append(random.normal(0, 1, i))
        data["Коши"].append(random.standard_cauchy(i))
        data["Лапласа"].append(random.laplace(0, 1/np.sqrt(2), i))
        data["Пуассона"].append(random.poisson(10, i))
        data["Равномерное"].append(random.uniform(-np.sqrt(3), np.sqrt(3), i))

    for name in data:
        paint_distribution(name, data[name], sample_sizes)

if __name__ == "__main__":
    main()