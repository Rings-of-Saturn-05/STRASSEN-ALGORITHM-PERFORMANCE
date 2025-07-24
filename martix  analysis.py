#!/usr/bin/env python3
"""
Strassen Algorithm Performance Visualizer
Reads benchmark data and creates comprehensive visualizations
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
import os
import sys

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_benchmark_data(filename="benchmark_results.txt"):
    """Load benchmark data from CSV file"""
    try:
        if not os.path.exists(filename):
            print(f"Error: {filename} not found!")
            print("Please run the C program first to generate benchmark data.")
            return None
        
        df = pd.read_csv(filename)
        print(f"Loaded {len(df)} benchmark results")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_performance_comparison(df):
    """Create side-by-side performance comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Execution Time Comparison
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, df['Naive_Time'], width, 
                   label='Naive O(n³)', color='#FF6B6B', alpha=0.8)
    bars2 = ax1.bar(x + width/2, df['Strassen_Time'], width,
                   label='Strassen O(n^2.807)', color='#4ECDC4', alpha=0.8)
    
    ax1.set_xlabel('Matrix Size')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Execution Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{size}×{size}' for size in df['Size']], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}s', ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}s', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Speedup Analysis
    ax2.plot(df['Size'], df['Speedup'], 'o-', linewidth=3, markersize=8, 
             color='#45B7D1', label='Actual Speedup')
    
    # Theoretical speedup line
    theoretical_speedup = [(size/50)**(3 - np.log(7)/np.log(2)) for size in df['Size']]
    ax2.plot(df['Size'], theoretical_speedup, '--', linewidth=2, 
             color='#96CEB4', label='Theoretical Speedup', alpha=0.7)
    
    ax2.set_xlabel('Matrix Size')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Strassen Speedup vs Matrix Size')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add speedup annotations
    for i, (size, speedup) in enumerate(zip(df['Size'], df['Speedup'])):
        if i % 2 == 0:  # Annotate every other point to avoid crowding
            ax2.annotate(f'{speedup:.1f}×', 
                        (size, speedup), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_complexity_analysis(df):
    """Create complexity analysis visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Log-scale complexity comparison
    sizes = df['Size']
    naive_times = df['Naive_Time']
    strassen_times = df['Strassen_Time']
    
    ax1.loglog(sizes, naive_times, 'o-', linewidth=2, markersize=6, 
               label='Naive (measured)', color='#FF6B6B')
    ax1.loglog(sizes, strassen_times, 's-', linewidth=2, markersize=6,
               label='Strassen (measured)', color='#4ECDC4')
    
    # Theoretical complexity lines
    n3_theoretical = naive_times[0] * (sizes / sizes.iloc[0])**3
    n_log7_theoretical = strassen_times[0] * (sizes / sizes.iloc[0])**(np.log(7)/np.log(2))
    
    ax1.loglog(sizes, n3_theoretical, '--', alpha=0.7, color='#FF6B6B',
               label='O(n³) theoretical')
    ax1.loglog(sizes, n_log7_theoretical, '--', alpha=0.7, color='#4ECDC4',
               label='O(n^2.807) theoretical')
    
    ax1.set_xlabel('Matrix Size (n)')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Algorithmic Complexity Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Efficiency analysis
    efficiency = []
    for i in range(len(df)):
        if i == 0:
            efficiency.append(100)
        else:
            expected_ratio = (df['Size'].iloc[i] / df['Size'].iloc[0])**(np.log(7)/np.log(2) - 3)
            actual_ratio = df['Strassen_Time'].iloc[0] / df['Strassen_Time'].iloc[i] * df['Naive_Time'].iloc[i] / df['Naive_Time'].iloc[0]
            efficiency.append(min(100, actual_ratio / expected_ratio * 100))
    
    bars = ax2.bar(range(len(df)), efficiency, color='#FFD93D', alpha=0.8)
    ax2.set_xlabel('Matrix Size')
    ax2.set_ylabel('Algorithm Efficiency (%)')
    ax2.set_title('Strassen Algorithm Efficiency')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels([f'{size}×{size}' for size in df['Size']], rotation=45)
    ax2.set_ylim(0, 110)
    ax2.grid(True, alpha=0.3)
    
    # Add efficiency labels
    for i, (bar, eff) in enumerate(zip(bars, efficiency)):
        ax2.text(bar.get_x() + bar.get_width()/2., eff + 2,
                f'{eff:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_crossover_analysis(df):
    """Analyze and visualize the crossover point"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create more detailed interpolation for smooth curves
    from scipy.interpolate import interp1d
    
    sizes_interp = np.linspace(df['Size'].min(), df['Size'].max(), 100)
    
    # Interpolate the data
    f_naive = interp1d(df['Size'], df['Naive_Time'], kind='cubic')
    f_strassen = interp1d(df['Size'], df['Strassen_Time'], kind='cubic')
    
    naive_interp = f_naive(sizes_interp)
    strassen_interp = f_strassen(sizes_interp)
    
    # Plot smooth curves
    ax.plot(sizes_interp, naive_interp, linewidth=3, label='Naive Algorithm', color='#FF6B6B')
    ax.plot(sizes_interp, strassen_interp, linewidth=3, label='Strassen Algorithm', color='#4ECDC4')
    
    # Plot actual data points
    ax.scatter(df['Size'], df['Naive_Time'], s=60, color='#FF6B6B', zorder=5, alpha=0.8)
    ax.scatter(df['Size'], df['Strassen_Time'], s=60, color='#4ECDC4', zorder=5, alpha=0.8)
    
    # Find crossover point using both methods
    # Method 1: First point where Strassen becomes faster (discrete)
    crossover_candidates = df[df['Speedup'] >= 1.0]
    discrete_crossover = None
    if len(crossover_candidates) > 0:
        discrete_crossover = crossover_candidates.iloc[0]['Size']
    
    # Method 2: Interpolated crossover point (but only look after speedup > 0.8)
    # Find where curves actually cross, not just where they're closest
    valid_mask = (sizes_interp >= df['Size'].min()) & (strassen_interp < naive_interp)
    if np.any(valid_mask):
        # Find the last point where Strassen is slower, then the crossover is just after
        last_slower_idx = np.where(strassen_interp >= naive_interp)[0]
        first_faster_idx = np.where(strassen_interp < naive_interp)[0]
        
        if len(last_slower_idx) > 0 and len(first_faster_idx) > 0:
            # Find the transition point
            transition_indices = []
            for i in range(len(sizes_interp)-1):
                if (strassen_interp[i] >= naive_interp[i] and 
                    strassen_interp[i+1] < naive_interp[i+1]):
                    transition_indices.append(i)
            
            if transition_indices:
                crossover_idx = transition_indices[0]
                interpolated_crossover = sizes_interp[crossover_idx]
                crossover_time = (naive_interp[crossover_idx] + strassen_interp[crossover_idx]) / 2
            else:
                # Fallback to discrete method
                interpolated_crossover = discrete_crossover
                if discrete_crossover:
                    crossover_time = df[df['Size'] == discrete_crossover]['Naive_Time'].iloc[0]
                else:
                    crossover_time = 0
        else:
            interpolated_crossover = discrete_crossover
            if discrete_crossover:
                crossover_time = df[df['Size'] == discrete_crossover]['Naive_Time'].iloc[0]
            else:
                crossover_time = 0
    else:
        interpolated_crossover = discrete_crossover
        if discrete_crossover:
            crossover_time = df[df['Size'] == discrete_crossover]['Naive_Time'].iloc[0]
        else:
            crossover_time = 0
    
    # Use the more reliable estimate
    if discrete_crossover and interpolated_crossover:
        if abs(discrete_crossover - interpolated_crossover) < discrete_crossover * 0.5:
            # If they're close, use interpolated
            crossover_size = interpolated_crossover
        else:
            # If they're far apart, trust the discrete method more
            crossover_size = discrete_crossover
            crossover_time = df[df['Size'] == discrete_crossover]['Naive_Time'].iloc[0]
    elif discrete_crossover:
        crossover_size = discrete_crossover
        crossover_time = df[df['Size'] == discrete_crossover]['Naive_Time'].iloc[0]
    else:
        # No crossover found
        crossover_size = None
        crossover_time = None
    
    # Mark crossover point if found
    if crossover_size:
        ax.plot(crossover_size, crossover_time, 'ro', markersize=12, 
                label=f'Crossover Point (~{crossover_size:.0f}×{crossover_size:.0f})')
        ax.axvline(x=crossover_size, color='red', linestyle='--', alpha=0.5)
        
        # Add annotations
        ax.annotate(f'Crossover at ~{crossover_size:.0f}×{crossover_size:.0f}\nTime: {crossover_time:.4f}s\nSpeedup ≥ 1.0x',
                    xy=(crossover_size, crossover_time),
                    xytext=(crossover_size + (df['Size'].max() - df['Size'].min()) * 0.1, 
                           crossover_time + crossover_time*0.5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=11, ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    else:
        # Add text indicating no crossover found
        ax.text(0.7, 0.7, 'No crossover point found\nin tested range', 
                transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
    
    ax.set_xlabel('Matrix Size (n×n)')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Algorithm Crossover Analysis')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    return fig
    
    return fig

def create_memory_complexity_info():
    """Create an informational plot about memory complexity"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sizes = [50, 100, 200, 300, 400, 500]
    naive_memory = [3 * s**2 * 8 / 1024**2 for s in sizes]  # 3 matrices, 8 bytes per double, MB
    strassen_memory = [20 * s**2 * 8 / 1024**2 for s in sizes]  # Approximate additional memory, MB
    
    ax.plot(sizes, naive_memory, 'o-', linewidth=2, label='Naive Algorithm', color='#FF6B6B')
    ax.plot(sizes, strassen_memory, 's-', linewidth=2, label='Strassen Algorithm', color='#4ECDC4')
    
    ax.set_xlabel('Matrix Size (n×n)')
    ax.set_ylabel('Memory Usage (MB)')
    ax.set_title('Memory Complexity Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    ax.text(300, 50, 'Strassen uses more memory\ndue to recursive structure\nand temporary matrices', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7),
            fontsize=10)
    
    return fig

def generate_summary_report(df):
    """Generate a text summary of the benchmark results"""
    print("\n" + "="*60)
    print("           STRASSEN ALGORITHM PERFORMANCE REPORT")
    print("="*60)
    
    print(f"\n📊 BENCHMARK SUMMARY:")
    print(f"   • Tested matrix sizes: {df['Size'].min()}×{df['Size'].min()} to {df['Size'].max()}×{df['Size'].max()}")
    print(f"   • Number of test cases: {len(df)}")
    print(f"   • Best speedup achieved: {df['Speedup'].max():.2f}× (at {df.loc[df['Speedup'].idxmax(), 'Size']}×{df.loc[df['Speedup'].idxmax(), 'Size']})")
    
    print(f"\n⚡ PERFORMANCE HIGHLIGHTS:")
    fastest_strassen = df.loc[df['Strassen_Time'].idxmin()]
    slowest_naive = df.loc[df['Naive_Time'].idxmax()]
    
    print(f"   • Fastest Strassen: {fastest_strassen['Strassen_Time']:.6f}s ({fastest_strassen['Size']}×{fastest_strassen['Size']})")
    print(f"   • Slowest Naive: {slowest_naive['Naive_Time']:.6f}s ({slowest_naive['Size']}×{slowest_naive['Size']})")
    print(f"   • Maximum time saved: {slowest_naive['Naive_Time'] - slowest_naive['Strassen_Time']:.6f}s")
    
    print(f"\n🎯 CROSSOVER ANALYSIS:")
    crossover_candidates = df[df['Speedup'] >= 1.0]
    if len(crossover_candidates) > 0:
        crossover_size = crossover_candidates.iloc[0]['Size']
        print(f"   • Crossover point: ~{crossover_size}×{crossover_size} matrices")
        print(f"   • Strassen becomes faster for matrices larger than {crossover_size}×{crossover_size}")
    else:
        print("   • Crossover point not reached in tested range")
        print("   • Consider testing larger matrices")
    
    print(f"\n📈 COMPLEXITY VERIFICATION:")
    print(f"   • Theoretical Strassen complexity: O(n^{np.log(7)/np.log(2):.3f})")
    print(f"   • Theoretical Naive complexity: O(n^3)")
    print(f"   • Complexity ratio: {3 / (np.log(7)/np.log(2)):.3f}")

def main():
    """Main visualization function"""
    print("Strassen Algorithm Performance Visualizer")
    print("=" * 45)
    
    # Load benchmark data
    df = load_benchmark_data()
    if df is None:
        return
    
    # Generate summary report
    generate_summary_report(df)
    
    print("\n🎨 Generating visualizations...")
    
    # Create visualizations
    # fig1 = create_performance_comparison(df)
    # fig1.suptitle('Strassen vs Naive: Performance Comparison', fontsize=16, y=1.02)
    
    # fig2 = create_complexity_analysis(df)
    # fig2.suptitle('Algorithmic Complexity Analysis', fontsize=16, y=1.02)
    
    fig3 = create_crossover_analysis(df)
    
    # fig4 = create_memory_complexity_info()
    
    # Save plots
    try:
        # fig1.savefig('strassen_performance_comparison.png', dpi=300, bbox_inches='tight')
        # fig2.savefig('strassen_complexity_analysis.png', dpi=300, bbox_inches='tight')
        fig3.savefig('strassen_crossover_analysis.png', dpi=300, bbox_inches='tight')
        # fig4.savefig('strassen_memory_analysis.png', dpi=300, bbox_inches='tight')
        
        print("\n✅ Visualizations saved:")
        # print("   • strassen_performance_comparison.png")
        # print("   • strassen_complexity_analysis.png") 
        print("   • strassen_crossover_analysis.png")
        # print("   • strassen_memory_analysis.png")
        
    except Exception as e:
        print(f"\n❌ Error saving plots: {e}")
    
    # Show plots
    plt.show()
    
    print("\n📋 Next steps:")
    print("   • Run with larger matrices for better crossover analysis")
    print("   • Compare with other fast matrix multiplication algorithms")
    print("   • Test on different hardware configurations")

if __name__ == "__main__":
    main()