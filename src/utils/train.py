import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

def train_epoch(model, loader, criterion, optimizer, device, gradient_clip=None, print_interval=10):
    """
    Funkcja trenująca model przez jedną epokę.
    
    Args:
        model: Model do trenowania.
        loader: DataLoader dla zbioru treningowego.
        criterion: Funkcja straty.
        optimizer: Optymalizator.
        device: Urządzenie (cpu/gpu).
        gradient_clip: Maksymalna norma gradientów (jeśli None - brak clippingu).
        print_interval: Co ile batchy wypisywać postępy.
        
    Returns:
        avg_loss: Średnia strata dla epoki.
        accuracy: Dokładność w trakcie treningu.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    total_batches = len(loader)
    # Użycie tqdm do wizualizacji postępu
    for batch_idx, (inputs, targets) in enumerate(tqdm(loader, desc="Trening", leave=False), 1):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def validate_epoch(model, loader, criterion, device, print_interval=10):
    """
    Funkcja walidująca model.
    
    Args:
        model: Model do walidacji.
        loader: DataLoader dla zbioru walidacyjnego.
        criterion: Funkcja straty.
        device: Urządzenie (cpu/gpu).
        print_interval: Co ile batchy wypisywać postęp walidacji.
        
    Returns:
        avg_loss: Średnia strata walidacyjna.
        accuracy: Dokładność walidacji.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(loader, desc="Walidacja", leave=False), 1):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx % print_interval == 0) or (batch_idx == total_batches):
                tqdm.write(f"Walidacja: Batch {batch_idx}/{total_batches}, Loss: {loss.item():.4f}")

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def train_model(model,
                train_loader: DataLoader,
                val_loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                scheduler,
                device: torch.device,
                num_epochs: int = 25,
                gradient_clip: float = None,
                print_interval: int = 10,
                save_dir: str = './saved_models',
                save_best: bool = True,
                save_every_epoch: bool = False):
    """
    Funkcja do trenowania modelu wraz z walidacją.
    
    Args:
        model: Trenowany model.
        train_loader: DataLoader ze zbioru treningowego.
        val_loader: DataLoader ze zbioru walidacyjnego.
        criterion: Funkcja straty.
        optimizer: Optymalizator.
        scheduler: Harmonogram uczenia, np. ReduceLROnPlateau lub inny.
        device: Urządzenie (cpu/gpu).
        num_epochs: Liczba epok treningowych.
        gradient_clip: Maksymalna norma gradientów (opcjonalnie).
        print_interval: Co ile batchy wypisywać postępy.
        save_dir: Folder do zapisywania modeli.
        save_best: Jeśli True – zapisuje najlepszy model (wg dokładności walidacji).
        save_every_epoch: Jeśli True – zapisuje model po każdej epoce.
        
    Returns:
        Słownik zawierający historię strat i dokładności:
            {
                'train_loss': [...],
                'train_acc':  [...],
                'val_loss':   [...],
                'val_acc':    [...]
            }
    """
    os.makedirs(save_dir, exist_ok=True)
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} - Rozpoczynam trening:")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device,
                                              gradient_clip=gradient_clip, print_interval=print_interval)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # Walidacja
        print(f"Epoch {epoch+1} - Rozpoczynam walidację:")
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, print_interval=print_interval)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch+1}: Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if scheduler is not None:
            scheduler.step(val_loss) 

        # Zapis najlepszego modelu lub po każdej epoce
        checkpoint_info = f"epoch_{epoch+1}_valacc_{val_acc:.2f}.pth"
        if save_best and (val_acc > best_val_acc):
            best_val_acc = val_acc
            model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Najlepszy model zapisany: {model_path}")
        if save_every_epoch:
            epoch_model_path = os.path.join(save_dir, checkpoint_info)
            torch.save(model.state_dict(), epoch_model_path)
            print(f"Model zapisany po epoce: {epoch_model_path}")

    print("\nTrening zakończony.")
    return history
