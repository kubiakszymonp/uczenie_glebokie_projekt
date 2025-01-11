import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

def get_predictions(model, loader, device):
    """
    Funkcja generuje predykcje i zbiera prawdziwe etykiety dla całego zbioru danych.
    
    Parametry:
      - model: trenowany model (PyTorch)
      - loader: DataLoader zawierający dane testowe
      - device: urządzenie (np. cuda, cpu)
      
    Zwraca:
      - all_preds: lista predykcji modelu
      - all_targets: lista prawdziwych etykiet
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Testowanie", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    return all_preds, all_targets

def evaluate_model(model, test_loader, criterion, device, class_names):
    """
    Funkcja ocenia testowy model, oblicza stratę i dokładność, 
    generuje raport klasyfikacji oraz wyświetla macierz konfuzji.
    
    Parametry:
      - model: trenowany model (PyTorch)
      - test_loader: DataLoader dla danych testowych
      - criterion: funkcja straty (np. nn.CrossEntropyLoss())
      - device: urządzenie (np. cuda, cpu)
      - class_names: lista nazw klas (np. ['NORMAL', 'BACTERIAL', 'VIRAL'])
    """
    # Test model i obliczanie straty oraz dokładności
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Obliczanie metryk", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            # Obliczanie liczby poprawnych predykcji
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    test_loss = total_loss / total
    test_acc = correct / total
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    
    # Pobieranie predykcji i prawdziwych etykiet
    test_preds, test_targets = get_predictions(model, test_loader, device)
    
    # Raport klasyfikacji
    report = classification_report(test_targets, test_preds, target_names=class_names)
    print("\nRaport klasyfikacji:\n", report)
    
    # Obliczanie macierzy konfuzji
    cm = confusion_matrix(test_targets, test_preds)
    print("\nMacierz konfuzji:")
    print(cm)
    
    # Wizualizacja macierzy konfuzji
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.ylabel('Prawdziwa etykieta')
    plt.xlabel('Predykcja')
    plt.title('Macierz konfuzji')
    plt.show()
