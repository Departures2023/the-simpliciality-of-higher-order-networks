import matplotlib.pyplot as plt
import re
import numpy as np
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.font_manager import FontProperties

def extract_cumulative_data(filename):
    """Extract all data from cumulative_data.txt file"""
    data = {
        'expected_es': [],
        'local_cluster_coefficient_avg': [],
        'local_cluster_coefficient_std': [],
        'connected_components_avg': [],
        'connected_components_std': [],
        'simplicial_fraction_avg': [],
        'simplicial_fraction_std': [],
        'edit_simpliciality_avg': [],
        'edit_simpliciality_std': [],
        'cumulative_edit_simpliciality_diff': [],
        'face_edit_simpliciality_avg': [],
        'face_edit_simpliciality_std': [],
        'density_avg': [],
        'density_std': [],
        'degree_count_avg': [],
        'degree_count_median': [],
        'degree_assortativity_avg': [],
        'degree_assortativity_std': [],
        'num_node_avg': [],
        'num_edge_avg': [],
        'num_node_std': [],
        'num_edge_std': [],
        'num_node_median': [],
        'num_edge_median': [],
        'num_node_max': [],
        'num_edge_max': [],
        'num_node_min': [],
        'num_edge_min': [],
        'evaluation_time_avg': [],
        'evaluation_time_std': [],
        'graph_generation_time_avg': [],
        'graph_generation_time_std': []
    }
    
    with open(filename, 'r') as f:
        content = f.read()
        
    # Split content by the separator lines
    blocks = content.split('=' * 50)
    
    for block in blocks:
        if 'expected_es:' in block:
            # Extract all values using regex
            for key in data.keys():
                if key == 'expected_es':
                    pattern = r'expected_es:\s*([\d.]+)'
                else:
                    pattern = rf'{key}:\s*([\d.-]+)'
                
                match = re.search(pattern, block)
                if match:
                    try:
                        value = float(match.group(1))
                        data[key].append(value)
                    except ValueError:
                        data[key].append(None)
                else:
                    data[key].append(None)
    
    return data

def create_plots(data):
    """Create all the requested plots"""
    
    # Set up the plotting style
    plt.style.use('default')
    fig_size = (10, 6)
    
    expected_es = np.array(data['expected_es'])
    
    # 1. Local cluster coefficient vs es
    plt.figure(figsize=fig_size)
    plt.errorbar(expected_es, data['local_cluster_coefficient_avg'], 
                yerr=data['local_cluster_coefficient_std'], 
                marker='o', capsize=5, capthick=2, label='Average ± Std')
    plt.xlabel('Expected ES')
    plt.ylabel('Local Cluster Coefficient')
    plt.title('Local Cluster Coefficient vs Expected ES')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('local_cluster_coefficient_vs_es.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Connected components vs es
    plt.figure(figsize=fig_size)
    plt.errorbar(expected_es, data['connected_components_avg'], 
                yerr=data['connected_components_std'], 
                marker='o', capsize=5, capthick=2, label='Average ± Std')
    plt.xlabel('Expected ES')
    plt.ylabel('Connected Components')
    plt.title('Connected Components vs Expected ES')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('connected_components_vs_es.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Simplicial fraction vs es
    plt.figure(figsize=fig_size)
    plt.errorbar(expected_es, data['simplicial_fraction_avg'], 
                yerr=data['simplicial_fraction_std'], 
                marker='o', capsize=5, capthick=2, label='Average ± Std')
    plt.xlabel('Expected ES')
    plt.ylabel('Simplicial Fraction')
    plt.title('Simplicial Fraction vs Expected ES')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('simplicial_fraction_vs_es.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Edit simpliciality vs es
    plt.figure(figsize=fig_size)
    plt.errorbar(expected_es, data['edit_simpliciality_avg'], 
                yerr=data['edit_simpliciality_std'], 
                marker='o', capsize=5, capthick=2, label='Average ± Std')
    plt.xlabel('Expected ES')
    plt.ylabel('Edit Simpliciality')
    plt.title('Edit Simpliciality vs Expected ES')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('edit_simpliciality_vs_es.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Face edit simpliciality vs es
    plt.figure(figsize=fig_size)
    plt.errorbar(expected_es, data['face_edit_simpliciality_avg'], 
                yerr=data['face_edit_simpliciality_std'], 
                marker='o', capsize=5, capthick=2, label='Average ± Std')
    plt.xlabel('Expected ES')
    plt.ylabel('Face Edit Simpliciality')
    plt.title('Face Edit Simpliciality vs Expected ES')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('face_edit_simpliciality_vs_es.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Density vs es
    plt.figure(figsize=fig_size)
    plt.errorbar(expected_es, data['density_avg'], 
                yerr=data['density_std'], 
                marker='o', capsize=5, capthick=2, label='Average ± Std')
    plt.xlabel('Expected ES')
    plt.ylabel('Density')
    plt.title('Density vs Expected ES')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('density_vs_es.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. Degree count vs es (avg and median)
    plt.figure(figsize=fig_size)
    plt.plot(expected_es, data['degree_count_avg'], 'o-', label='Average', linewidth=2)
    plt.plot(expected_es, data['degree_count_median'], 's-', label='Median', linewidth=2)
    plt.xlabel('Expected ES')
    plt.ylabel('Degree Count')
    plt.title('Degree Count vs Expected ES')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('degree_count_vs_es.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 8. Number of nodes vs es (comprehensive)
    plt.figure(figsize=fig_size)
    plt.errorbar(expected_es, data['num_node_avg'], 
                yerr=data['num_node_std'], 
                marker='o', capsize=5, capthick=2, label='Average ± Std')
    plt.plot(expected_es, data['num_node_median'], 's-', label='Median', alpha=0.7)
    plt.plot(expected_es, data['num_node_max'], '^-', label='Maximum', alpha=0.7)
    plt.plot(expected_es, data['num_node_min'], 'v-', label='Minimum', alpha=0.7)
    plt.xlabel('Expected ES')
    plt.ylabel('Number of Nodes')
    plt.title('Number of Nodes vs Expected ES')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('num_node_vs_es.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 9. Number of edges vs es (comprehensive)
    plt.figure(figsize=fig_size)
    plt.errorbar(expected_es, data['num_edge_avg'], 
                yerr=data['num_edge_std'], 
                marker='o', capsize=5, capthick=2, label='Average ± Std')
    plt.plot(expected_es, data['num_edge_median'], 's-', label='Median', alpha=0.7)
    plt.plot(expected_es, data['num_edge_max'], '^-', label='Maximum', alpha=0.7)
    plt.plot(expected_es, data['num_edge_min'], 'v-', label='Minimum', alpha=0.7)
    plt.xlabel('Expected ES')
    plt.ylabel('Number of Edges')
    plt.title('Number of Edges vs Expected ES')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('num_edge_vs_es.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 10. Evaluation time vs es
    plt.figure(figsize=fig_size)
    plt.errorbar(expected_es, data['evaluation_time_avg'], 
                yerr=data['evaluation_time_std'], 
                marker='o', capsize=5, capthick=2, label='Average ± Std')
    plt.xlabel('Expected ES')
    plt.ylabel('Evaluation Time (s)')
    plt.title('Evaluation Time vs Expected ES')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('evaluation_time_vs_es.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 11. Graph generation time vs es
    plt.figure(figsize=fig_size)
    plt.errorbar(expected_es, data['graph_generation_time_avg'], 
                yerr=data['graph_generation_time_std'], 
                marker='o', capsize=5, capthick=2, label='Average ± Std')
    plt.xlabel('Expected ES')
    plt.ylabel('Graph Generation Time (s)')
    plt.title('Graph Generation Time vs Expected ES')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graph_generation_time_vs_es.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 12. Cumulative edit simpliciality diff vs es
    plt.figure(figsize=fig_size)
    plt.plot(expected_es, data['cumulative_edit_simpliciality_diff'], 'o-', linewidth=2)
    plt.xlabel('Expected ES')
    plt.ylabel('Cumulative Edit Simpliciality Difference')
    plt.title('Cumulative Edit Simpliciality Difference vs Expected ES')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cumulative_edit_simpliciality_diff_vs_es.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 13. Degree assortativity vs es
    plt.figure(figsize=fig_size)
    plt.errorbar(expected_es, data['degree_assortativity_avg'], 
                yerr=data['degree_assortativity_std'], 
                marker='o', capsize=5, capthick=2, label='Average ± Std')
    plt.xlabel('Expected ES')
    plt.ylabel('Degree Assortativity')
    plt.title('Degree Assortativity vs Expected ES')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('degree_assortativity_vs_es.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_plot(data):
    """Create a summary plot with multiple subplots"""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Summary: All Metrics vs Expected ES', fontsize=16)
    
    expected_es = np.array(data['expected_es'])
    
    # Define the plots for the summary
    plots_config = [
        ('local_cluster_coefficient_avg', 'local_cluster_coefficient_std', 'Local Cluster Coefficient'),
        ('connected_components_avg', 'connected_components_std', 'Connected Components'),
        ('simplicial_fraction_avg', 'simplicial_fraction_std', 'Simplicial Fraction'),
        ('edit_simpliciality_avg', 'edit_simpliciality_std', 'Edit Simpliciality'),
        ('face_edit_simpliciality_avg', 'face_edit_simpliciality_std', 'Face Edit Simpliciality'),
        ('density_avg', 'density_std', 'Density'),
        ('degree_count_avg', 'degree_count_median', 'Degree Count'),
        ('num_edge_avg', 'num_edge_std', 'Number of Edges'),
        ('evaluation_time_avg', 'evaluation_time_std', 'Evaluation Time'),
        ('graph_generation_time_avg', 'graph_generation_time_std', 'Graph Generation Time'),
        ('degree_assortativity_avg', 'degree_assortativity_std', 'Degree Assortativity'),
        ('cumulative_edit_simpliciality_diff', None, 'Cumulative Edit Simpliciality Diff')
    ]
    
    for i, (avg_key, std_key, title) in enumerate(plots_config):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        if std_key and std_key in data:
            if 'median' in std_key:
                # Special case for degree count (avg vs median)
                ax.plot(expected_es, data[avg_key], 'o-', label='Average', linewidth=2)
                ax.plot(expected_es, data[std_key], 's-', label='Median', linewidth=2)
                ax.legend(fontsize=8)
            else:
                # Standard error bar plot
                ax.errorbar(expected_es, data[avg_key], 
                           yerr=data[std_key], 
                           marker='o', capsize=3, capthick=1)
        else:
            # Single line plot
            ax.plot(expected_es, data[avg_key], 'o-', linewidth=2)
        
        ax.set_xlabel('Expected ES', fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.savefig('summary_all_metrics_vs_es.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main execution
if __name__ == "__main__":
    # File path
    file_path = r'experiment_result\model_generation_es\cumulative_data.txt'
    
    # Extract data
    print("Extracting data from cumulative_data.txt...")
    data = extract_cumulative_data(file_path)
    
    # Print some basic info
    print(f"Found {len(data['expected_es'])} data points")
    print(f"Expected ES range: {min(data['expected_es']):.3f} to {max(data['expected_es']):.3f}")
    
    # Create individual plots
    print("Creating individual plots...")
    create_plots(data)
    
    # Create summary plot
    print("Creating summary plot...")
    create_summary_plot(data)
    
    print("All plots have been generated and saved!")