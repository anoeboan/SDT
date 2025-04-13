import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
import os

from SoftDecisionTree import SDT

class CustomDataset(Dataset):
    def __init__(self, data, targets):
        """
        data: Feature data (numpy array or pandas dataframe)
        targets:  target labela (numpy array)
        """
        self.data = torch.tensor(data, dtype=torch.float32)  
        self.targets = torch.tensor(targets, dtype=torch.long) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def load_dataset(dataset_name, batch_size):
    if dataset_name == "MNIST":
        data_dir = "../Dataset/mnist"
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_loader = DataLoader(
            datasets.MNIST(data_dir, train=True, download=True, transform=transformer),
            batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            datasets.MNIST(data_dir, train=False, transform=transformer),
            batch_size=batch_size, shuffle=False
        )
    elif dataset_name == "FashionMNIST":
        data_dir = "../Dataset/fashion_mnist"
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_loader = DataLoader(
            datasets.FashionMNIST(data_dir, train=True, download=True, transform=transformer),
            batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            datasets.FashionMNIST(data_dir, train=False, transform=transformer),
            batch_size=batch_size, shuffle=False
        )
    elif dataset_name == "CIFAR10":
        data_dir = "../Dataset/cifar10"
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        train_loader = DataLoader(
            datasets.CIFAR10(data_dir, train=True, download=True, transform=transformer),
            batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            datasets.CIFAR10(data_dir, train=False, transform=transformer),
            batch_size=batch_size, shuffle=False
        )
    elif dataset_name == "SVHN":
        data_dir = "../Dataset/svhn"
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_loader = DataLoader(
            datasets.SVHN(data_dir, split='train', download=True, transform=transformer),
            batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            datasets.SVHN(data_dir, split='test', download=True, transform=transformer),
            batch_size=batch_size, shuffle=False
        )
    else:
        raise ValueError("Unsupported dataset")

    return train_loader, test_loader

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    depths = [3, 5, 7]
    datasets_to_test = ["FashionMNIST", "MNIST", "CIFAR10", "SVHN"] 
    batch_size = 128
    epochs = 40
    
    depth_results = []
    dataset_results = []
    
    log_file = open("results/training_log.txt", "w")
    
    for dataset_name in datasets_to_test:
        train_loader, test_loader = load_dataset(dataset_name, batch_size)
        
        for depth in depths:
            log_file.write(f"Current dataset: {dataset_name} | Current depth: {depth}\n")
            print(f"Current dataset: {dataset_name} | Current depth: {depth}")
            if dataset_name in ["MNIST", "FashionMNIST"]:
                input_dim = 28 * 28  # 784
            elif dataset_name in ["CIFAR10", "SVHN"]:
                input_dim = 32 * 32 * 3  # 3072
            else:
                raise ValueError("Unsupported dataset")

            tree = SDT(input_dim, 10, depth, 1e-3)
            optimizer = torch.optim.Adam(tree.parameters(), lr=1e-3, weight_decay=5e-4)
            criterion = nn.CrossEntropyLoss()
            
            training_loss_list = []
            testing_acc_list = []
            auc_list = []
            
            for epoch in range(epochs):
                tree.train()
                epoch_loss = 0
                for data, target in train_loader:
                    data, target = data.to(tree.device), target.to(tree.device)
                    
                    # target_onehot = onehot_coding(target, tree.device, 10)
                    output, penalty = tree.forward(data, is_train=True)
                    loss = criterion(output, target.view(-1)) + penalty
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                training_loss_list.append(epoch_loss / len(train_loader))
                
                tree.eval()
                correct = 0
                all_preds = []
                all_targets = []
                
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(tree.device), target.to(tree.device)
                        output = F.softmax(tree.forward(data), dim=1)
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        all_preds.extend(output.cpu().numpy())
                        all_targets.extend(target.cpu().numpy())
                
                accuracy = 100.0 * correct / len(test_loader.dataset)
                testing_acc_list.append(accuracy)

                all_preds = np.array(all_preds)
                all_targets = np.array(all_targets).astype(int)
                all_preds = F.softmax(torch.tensor(all_preds), dim=1).numpy()
                if np.isnan(all_preds).any():
                    print("Warning: NaN detected in all_preds!")
                    all_preds = np.nan_to_num(all_preds)
                all_preds = all_preds.reshape(-1, 10)

                auc = roc_auc_score(np.eye(10)[all_targets], all_preds, multi_class='ovr')
                auc_list.append(auc)

                log_file.write(f"Epoch {epoch} | Train Loss: {epoch_loss / len(train_loader):.4f} | Test Accuracy: {accuracy:.2f}% | AUC: {auc:.4f}\n")
                print(f"Epoch {epoch} | Train Loss: {epoch_loss / len(train_loader):.4f} | Test Accuracy: {accuracy:.2f}% | AUC: {auc:.4f}")
    
            depth_results.append([dataset_name, depth, training_loss_list[-1], testing_acc_list[-1], auc_list[-1]])
            dataset_results.append([dataset_name, depth, training_loss_list, testing_acc_list, auc_list])
    
    depth_df = pd.DataFrame(depth_results, columns=["Dataset", "Depth", "Final Train Loss", "Final Test Accuracy", "Final AUC"])
    depth_df.to_excel("results/depth_comparison.xlsx", index=False)
    
    dataset_df = pd.DataFrame(dataset_results, columns=["Dataset", "Depth", "Train Losses", "Test Accuracies", "AUC Scores"])
    dataset_df.to_excel("results/dataset_comparison.xlsx", index=False)
    
    for dataset_name in datasets_to_test:
        plt.figure(figsize=(10, 5))
        for depth in depths:
            subset = dataset_df[(dataset_df["Dataset"] == dataset_name) & (dataset_df["Depth"] == depth)]
            plt.plot(subset.iloc[0]["Train Losses"], label=f"Depth {depth} - Loss")
        plt.title(f"Loss Comparison ({dataset_name})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"results/{dataset_name}_loss.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        for depth in depths:
            subset = dataset_df[(dataset_df["Dataset"] == dataset_name) & (dataset_df["Depth"] == depth)]
            plt.plot(subset.iloc[0]["Test Accuracies"], label=f"Depth {depth} - Accuracy")
        plt.title(f"Test Accuracy Comparison ({dataset_name})")
        plt.xlabel("Epoch")
        plt.ylabel("Test Accuracy (%)")
        plt.legend()
        plt.savefig(f"results/{dataset_name}_accuracy.png")
        plt.close()
    
    log_file.close()
    print("Results saved to Excel and images. Training log saved to 'results/training_log.txt'.")