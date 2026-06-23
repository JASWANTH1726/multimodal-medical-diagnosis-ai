"""Evaluation script"""

import torch
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


def evaluate(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    predictions = []
    ground_truth = []
    confidences = []
    
    with torch.no_grad():
        for images, texts, labels in test_loader:
            images = images.to(device)
            logits = model(images, texts)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            ground_truth.extend(labels.numpy())
            confidences.extend(probs.max(dim=1).values.cpu().numpy())
    
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    confidences = np.array(confidences)
    
    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, average='weighted', zero_division=0)
    recall = recall_score(ground_truth, predictions, average='weighted', zero_division=0)
    f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Avg Confidence: {confidences.mean():.4f}")
    print("="*50 + "\n")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_confidence': confidences.mean()
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate medical diagnosis model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to test data')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Loading model from: {args.model}")
    print(f"Test data from: {args.data}")


if __name__ == '__main__':
    main()
