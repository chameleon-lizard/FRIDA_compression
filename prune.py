import transformers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
from tqdm import tqdm
from datasets import load_dataset

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and prepare data
def load_and_prepare_data(dataset_name):
    dataset = load_dataset(dataset_name)
    all_data = []

    for language, ds in dataset.items():
        df = pd.DataFrame(ds)

        # Prepare positive samples (refusals)
        refusals = pd.concat([
            df[['refusal_1']].rename(columns={'refusal_1': 'text'}),
            df[['refusal_2']].rename(columns={'refusal_2': 'text'})
        ], ignore_index=True)
        refusals['label'] = 1
        refusals['language'] = language  # Add language column

        # Prepare negative samples (answers)
        non_refusals = pd.concat([
            df[['answer_1']].rename(columns={'answer_1': 'text'}),
            df[['answer_2']].rename(columns={'answer_2': 'text'}),
        ], ignore_index=True)
        non_refusals['label'] = 0
        non_refusals['language'] = language  # Add language column

        # Combine and shuffle
        data = pd.concat([refusals, non_refusals], ignore_index=True)

        # Ensure data is not empty before shuffling
        if data.empty:
            raise ValueError(f"Data for language {language} is empty or invalid.")

        data = data.sample(frac=1, random_state=42).reset_index(drop=True)

        # Handle missing values
        missing_count = data['text'].isna().sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} rows with missing 'text' detected for language {language}. Dropping these rows.")
        data = data.dropna(subset=['text'])

        all_data.append(data)

    # Combine data from all languages
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)  # Ensure multilingual shuffling
    return combined_data

# Tokenize test data
def tokenize_and_create_dataset(tokenizer, test_df):
    test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=512)

    # Convert to PyTorch Dataset
    class RefusalDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    test_dataset = RefusalDataset(test_encodings, test_df['label'].tolist())
    print(f'Test dataset size: {len(test_dataset)}')
    return test_dataset

# Function to compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=['Non-Refusal', 'Refusal'], output_dict=True)
    return {"accuracy": acc, "f1": report['Refusal']['f1-score'], "precision": report['Refusal']['precision'], "recall": report['Refusal']['recall']}


# Function to evaluate the model
def evaluate_model(model, dataloader):
    model.eval()
    model.to(device)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            # Ensure logits are in the correct shape
            logits = logits.cpu().numpy()
            all_preds.extend(logits)
            all_labels.extend(batch['labels'].cpu().numpy())
    
    eval_pred = transformers.EvalPrediction(predictions=np.array(all_preds), label_ids=np.array(all_labels))
    return compute_metrics(eval_pred)

def calculate_model_size_in_millions(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def prune_model_layers(model_path, dataset_name, f1_threshold=0.8):
    """
    Prunes layers of a Transformer model based on minimal impact to F1 score.

    Args:
        model_path (str): Path to the pretrained model.
        dataset_name (str): Name of the dataset from Hugging Face.
        f1_threshold (float): F1 score threshold to continue pruning.

    Returns:
        list: A list of indices of the layers pruned.
    """

    # Load data and split into train and test
    data = load_and_prepare_data(dataset_name)
    _, test_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])

    # Load model and tokenizer
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model.to(device)

    # Tokenize test data
    test_dataset = tokenize_and_create_dataset(tokenizer, test_df)

    # Define DataLoader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Initial evaluation
    initial_metrics = evaluate_model(model, test_loader)
    print(f'Initial model size: {calculate_model_size_in_millions(model)}M')
    print(f"Initial F1 Score: {initial_metrics['f1']}")

    # Pruning loop
    layers_pruned = []
    current_f1 = initial_metrics['f1']

    while current_f1 >= f1_threshold:
        min_impact_f1 = float('-inf')
        layer_to_remove = None

        # Load a new model for each pruning iteration
        pruned_model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)
        pruned_model.to(device)  # Move model to device

        # Evaluate each layer for pruning impact
        for layer_idx in tqdm(range(len(pruned_model.bert.encoder.layer) - len(layers_pruned))):
            # Create a new model for each layer evaluation
            temp_model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)
            temp_model.to(device)  # Move model to device

            for pruned_layer in layers_pruned:
                temp_model.bert.encoder.layer.pop(pruned_layer)

            # Remove the current layer
            temp_model.bert.encoder.layer.pop(layer_idx)

            # Evaluate the pruned model
            pruned_metrics = evaluate_model(temp_model, test_loader)
            pruned_f1 = pruned_metrics['f1']
            print(f'Pruned block {layer_idx}, F1: {pruned_f1}')

            # Check if this layer removal has the least impact on F1 score
            if pruned_f1 > min_impact_f1:
                min_impact_f1 = pruned_f1
                layer_to_remove = layer_idx

        if layer_to_remove is not None and min_impact_f1 > f1_threshold:
            layers_pruned.append(layer_to_remove)
            current_f1 = min_impact_f1
            print(f"Removed block {layer_to_remove}, new F1 Score: {current_f1}, new model size: {calculate_model_size_in_millions(temp_model)}M")
        else:
            break

    print(f"Final blocks pruned: {layers_pruned}")
    print(f"Final F1 Score: {current_f1}")
    return layers_pruned

if __name__ == '__main__':
    # Example usage
    dataset_name = 'chameleon-lizard/multilingual_refusals'
    model_path = './refusal_classifier_small/'
    pruned_layers = prune_model_layers(model_path, dataset_name, f1_threshold=0.8)
    print("Pruned Layers:", pruned_layers)
