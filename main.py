import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def calculate_chi2(sample, alpha=0.05):
    n = len(sample)
    
    mu_hat = np.mean(sample)
    sigma_hat = np.std(sample, ddof=0)
    
    k = int(np.round(1 + 3.322 * np.log10(n)))
    if k < 4:
        k = 4

    observed_freq, bin_edges = np.histogram(sample, bins=k)
    
    cdf_edges = bin_edges.copy()
    cdf_edges[0] = -np.inf
    cdf_edges[-1] = np.inf
    
    p_i = np.zeros(k)
    for i in range(k):
        p_i[i] = stats.norm.cdf(cdf_edges[i+1], loc=mu_hat, scale=sigma_hat) - \
                 stats.norm.cdf(cdf_edges[i], loc=mu_hat, scale=sigma_hat)
                 
    expected_freqs = n * p_i
    
    chi2_stat = np.sum((observed_freq - expected_freqs)**2 / expected_freqs)
    
    df = k - 1 
    if df > 0:
        critical_value = stats.chi2.ppf(1 - alpha, df)
    else:
        critical_value = 0.0
            
    return mu_hat, sigma_hat, chi2_stat, critical_value, df, bin_edges, observed_freq, expected_freqs


def plot_single_test(sample, dist_name, ax):
    mu, sigma, chi2, crit, df, bins, obs, exp = calculate_chi2(sample)
    
    plot_bins = bins.copy()
    plot_bins[0] = min(sample) - 0.5
    plot_bins[-1] = max(sample) + 0.5
    
    ax.hist(sample, bins=plot_bins, edgecolor='black', alpha=0.6, density=True, color='skyblue')
    
    x = np.linspace(plot_bins[0], plot_bins[-1], 1000)
    pdf = stats.norm.pdf(x, loc=mu, scale=sigma)
    ax.plot(x, pdf, 'r-', lw=2, label=f'N({mu:.2f}, {sigma:.2f})')
    
    for b in bins[1:-1]:
        ax.axvline(b, color='red', linestyle='--', alpha=0.5, lw=1)
        
    ax.set_title(f"{dist_name} (n={len(sample)})\nχ²_набл={chi2:.2f}, χ²_крит={crit:.2f}")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)


def main():
    random = np.random.default_rng(6)
    alpha = 0.05
    
    sample_norm = random.normal(0, 1, 100)
    sample_unif = random.uniform(-np.sqrt(3), np.sqrt(3), 20)
    sample_lapl = random.laplace(0, 1/np.sqrt(2), 20)
    
    test_cases = [
        ("Нормальное N(0,1)", sample_norm),
        ("Равномерное", sample_unif),
        ("Лапласа", sample_lapl)
    ]

    print("="*90)
    print("РЕЗУЛЬТАТЫ ПРОВЕРКИ ГИПОТЕЗЫ О НОРМАЛЬНОСТИ (α = 0.05)")
    print("="*90)
    print(f"{'Распределение':<20} | {'n':<5} | {'μ^':<5} | {'σ^':<5} | {'χ²_набл':<8} | {'χ²_крит':<8} | {'Результат':<15}")
    print("-" * 90)
    
    for name, sample in test_cases:
        mu, sigma, chi2, crit_val, df, _, _, _ = calculate_chi2(sample, alpha)
        
        decision = "Принимается" if chi2 < crit_val else "ОТКЛОНЯЕТСЯ"
        
        print(f"{name:<20} | {len(sample):<5} | {mu:>5.2f} | {sigma:>5.2f} | {chi2:>8.2f} | {crit_val:>8.2f} | {decision:<15}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, (name, sample) in enumerate(test_cases):
        plot_single_test(sample, name, axes[i])
    
    plt.suptitle("Проверка гипотез о нормальности", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()