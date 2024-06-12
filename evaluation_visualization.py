import matplotlib.pyplot as plt

def plot_evaluation_results(results):
    accuracies = [result['accuracy'] for result in results]
    datasets = [result['dataset'] for result in results]

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.bar(datasets, accuracies, color='b', alpha=0.5)
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Models on Different Datasets')
    plt.ylim(0.9, 1.0)  # Set y-axis limits to emphasize differences
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()