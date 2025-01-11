import matplotlib.pyplot as plt

def plot_training_validation(train_losses, val_losses, train_accuracies, val_accuracies, epochs=None):
    """
    Plots training and validation loss and accuracy side by side.

    Parameters:
    - train_losses: list or array of training loss values.
    - val_losses: list or array of validation loss values.
    - train_accuracies: list or array of training accuracy values.
    - val_accuracies: list or array of validation accuracy values.
    - epochs: sequence of epoch numbers. If None, epochs 1..N are used.
    """
    
    if epochs is None:
        epochs = range(1, len(train_losses) + 1)

    # Unified style parameters
    train_style = 'bo-'
    val_style   = 'ro-'
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Losses
    axes[0].plot(epochs, train_losses, train_style, label='Train')
    axes[0].plot(epochs, val_losses, val_style, label='Validation')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Plot Accuracies
    axes[1].plot(epochs, train_accuracies, train_style, label='Train')
    axes[1].plot(epochs, val_accuracies, val_style, label='Validation')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_single_metric(epochs, train_metric, val_metric, title, ylabel):
    """
    Plots a single metric for both training and validation.

    Parameters:
    - epochs: sequence of epoch numbers.
    - train_metric: list or array of training metric values.
    - val_metric: list or array of validation metric values.
    - title: title for the plot.
    - ylabel: label for the y-axis.
    """
    
    # Unified style for both lines
    train_style = 'bo-'
    val_style   = 'ro-'
    
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_metric, train_style, label='Train')
    plt.plot(epochs, val_metric, val_style, label='Validation')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()