import matplotlib.pyplot as plt

def plot_evaluation_results(results):
    datasets = [result['dataset'] for result in results]
    accuracies = [result['accuracy'] for result in results]
    precisions = [result['precision'] for result in results]
    recalls = [result['recall'] for result in results]
    f1_scores = [result['f1-score'] for result in results]

    # Tworzenie wykresu dok�adno�ci (Accuracy)
    plt.figure(figsize=(10, 5))
    bars = plt.bar(datasets, accuracies, color='blue', alpha=0.7)
    plt.title('Accuracy Comparison')
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Dodawanie etykiet tekstowych
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{accuracy:.10f}', 
                 ha='center', va='bottom', fontsize=8, color='black')
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')

    # Tworzenie wykresu precyzji (Precision)
    plt.figure(figsize=(10, 5))
    bars = plt.bar(datasets, precisions, color='green', alpha=0.7)
    plt.title('Precision Comparison')
    plt.xlabel('Dataset')
    plt.ylabel('Precision')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Dodawanie etykiet tekstowych
    for bar, precision in zip(bars, precisions):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{precision:.10f}', 
                 ha='center', va='bottom', fontsize=8, color='black')
    
    plt.tight_layout()
    plt.savefig('precision_comparison.png')

    # Tworzenie wykresu odzysku (Recall)
    plt.figure(figsize=(10, 5))
    bars = plt.bar(datasets, recalls, color='orange', alpha=0.7)
    plt.title('Recall Comparison')
    plt.xlabel('Dataset')
    plt.ylabel('Recall')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Dodawanie etykiet tekstowych
    for bar, recall in zip(bars, recalls):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{recall:.10f}', 
                 ha='center', va='bottom', fontsize=8, color='black')
    
    plt.tight_layout()
    plt.savefig('recall_comparison.png')

    # Tworzenie wykresu F1-score
    plt.figure(figsize=(10, 5))
    bars = plt.bar(datasets, f1_scores, color='red', alpha=0.7)
    plt.title('F1-score Comparison')
    plt.xlabel('Dataset')
    plt.ylabel('F1-score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Dodawanie etykiet tekstowych
    for bar, f1_score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{f1_score:.10f}', 
                 ha='center', va='bottom', fontsize=8, color='black')
    
    plt.tight_layout()
    plt.savefig('f1_score_comparison.png')

    plt.show()