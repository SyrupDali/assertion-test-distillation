#!/usr/bin/env python
"""
Reads multiple JSON metric files and generates grouped bar plots
to compare model performance across different configurations.
"""
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_metric_data(files: list) -> list:
    """Loads data from a list of JSON file paths."""
    all_data = []
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                all_data.append(json.load(f))
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            exit(1)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}")
            exit(1)
    return all_data

def create_comparison_plot(data_dict: dict, labels: list, title: str, output_filename: str):
    """
    Creates and saves a grouped bar chart from a dictionary of metric data.
    """
    metric_names = list(data_dict.keys())
    num_metrics = len(metric_names)
    num_models = len(labels)

    # --- MODIFIED: Adjust group spacing and figure size ---
    group_spacing_factor = 1.1 # Increase this value to add more space between groups
    bar_width = 1.0 / num_models # Keep bars relatively thin
    
    # Set the positions for each group of bars, now spaced out
    x = np.arange(num_metrics) * group_spacing_factor 

    # Increase the figure size to accommodate the wider spacing
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Calculate the offset for each bar within a group
    # This formula centers the group of bars over the x-tick
    offsets = np.arange(num_models) * bar_width - (bar_width * (num_models - 1) / 2)

    # Plot the bars for each model
    for i, label in enumerate(labels):
        model_scores = [data_dict[metric][i] for metric in metric_names]
        # Apply the offset to the spaced-out group positions
        rects = ax.bar(x + offsets[i], model_scores, bar_width, label=label)
        ax.bar_label(rects, padding=3, fmt='%.3f', fontsize=8)

    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Scores')
    ax.set_title(title)
    ax.set_xticks(x) # The tick marks are now at the center of each spaced-out group
    ax.set_xticklabels(metric_names)
    ax.legend()

    ax.set_ylim(0, 1.15) # Increased ylim slightly for more headroom
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    fig.tight_layout()
    plt.savefig(output_filename)
    print(f"✅ Plot saved to {output_filename}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare distillation model metrics from JSON files.")
    
    parser.add_argument(
        '--files', 
        nargs='+',
        required=True, 
        help='Space-separated paths to the JSON metric files.'
    )
    parser.add_argument(
        '--labels', 
        nargs='+',
        required=True, 
        help='Space-separated labels for the models, corresponding to the files.'
    )
    
    parser.add_argument(
        '--output_prefix',
        type=str,
        default='comparison',
        help='The prefix for the output plot image files.'
    )
    args = parser.parse_args()

    if len(args.files) != len(args.labels):
        raise ValueError("The number of provided files must match the number of provided labels.")

    all_metrics_data = load_metric_data(args.files)

    # Prepare data for Plot 1
    metrics_plot1 = ['precision', 'recall', 'f1', 'accuracy']
    data_plot1 = {
        metric.replace('_', ' ').capitalize(): [data.get(metric, 0) for data in all_metrics_data]
        for metric in metrics_plot1
    }

    # Prepare data for Plot 2
    metrics_plot2 = ['avg_similarity', 'avg_codebleu', 'avg_ast_validity', 'avg_token_accuracy']
    metric_name_map = {
        'avg_similarity': 'Seq. Similarity',
        'avg_codebleu': 'CodeBLEU',
        'avg_ast_validity': 'AST Validity',
        'avg_token_accuracy': 'Token Accuracy'
    }
    data_plot2 = {
        metric_name_map[metric]: [data.get(metric, 0) for data in all_metrics_data]
        for metric in metrics_plot2
    }

    # Generate the plots
    create_comparison_plot(
        data_dict=data_plot1,
        labels=args.labels,
        title='Comparison of Core Evaluation Metrics',
        output_filename=f"{args.output_prefix}_core_metrics.png"
    )
    
    create_comparison_plot(
        data_dict=data_plot2,
        labels=args.labels,
        title='Comparison of Code-Specific and Similarity Metrics',
        output_filename=f"{args.output_prefix}_code_metrics.png"
    )

if __name__ == '__main__':
    main()