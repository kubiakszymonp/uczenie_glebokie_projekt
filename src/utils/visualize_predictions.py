import torch
import matplotlib.pyplot as plt
import numpy as np


def visualize_predictions(model, dataloader, class_names, device, num_images=5):
    """
    Visualizes a specified number of images along with their true and predicted labels.

    Args:
        model (nn.Module): Trained PyTorch model.
        dataloader (DataLoader): DataLoader for the dataset to visualize.
        class_names (list): List of class names corresponding to model outputs.
        device (torch.device): Device to perform computations on.
        num_images (int): Number of images to visualize.
    """
    model.eval()
    images_shown = 0
    plt.figure(figsize=(15, 15))

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(inputs.size(0)):
                images_shown += 1
                ax = plt.subplot(int(np.ceil(num_images / 5)), 5, images_shown)
                ax.axis("off")
                ax.set_title(
                    f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}"
                )

                # Unnormalize the image for display
                img = inputs.cpu().data[i]
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(
                    3, 1, 1
                ) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                img = img.numpy()
                img = np.clip(img, 0, 1)  # Ensure pixel values are between 0 and 1
                plt.imshow(np.transpose(img, (1, 2, 0)))

                if images_shown == num_images:
                    plt.show()
                    return

    plt.show()
