import os
import subprocess
from trainer import train_classifier
from prune import prune_model_layers, evaluate_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd


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

        # Prepare negative samples (answers)
        non_refusals = pd.concat([
            df[['answer_1']].rename(columns={'answer_1': 'text'}),
            df[['answer_2']].rename(columns={'answer_2': 'text'}),
        ], ignore_index=True)
        non_refusals['label'] = 0

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


def run_training(model_name, output_dir, dataset_name, batch_size=64, epochs=1, lr=1e-4, seed=42, max_length=512, initial_train=False, f1_prune_threshold=0.8):
    """
    Trains a model using the train_classifier function from trainer.py.
    """
    # Create the models/ directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    output_dir = os.path.join('models', output_dir)

    if not initial_train:
        model_name = os.path.join('models', model_name)

    print(f"Training: Model={model_name}, Output={output_dir}")
    eval_results = train_classifier(
        dataset_name=dataset_name,
        model_name=model_name,
        output_dir=output_dir,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        seed=seed,
        f1_prune_threshold=f1_prune_threshold,
        max_length=max_length
    )
    print(f"Evaluation Results after Training: {eval_results}")
    return eval_results


def run_pruning(model_path, dataset_name, output_dir, f1_threshold=0.8):
    """
    Prunes a model using the prune_model_layers function from prune.py.
    """
    # Create the models/ directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    output_dir = os.path.join('models', output_dir)
    model_path = os.path.join('models', model_path)
    
    print(f"Pruning: Model={model_path}, Output={output_dir}")

    # Load data and split into train and test
    data = load_and_prepare_data(dataset_name)
    _, test_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Tokenize test data
    test_dataset = tokenize_and_create_dataset(tokenizer, test_df)
    # Define DataLoader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Initial evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    initial_metrics = evaluate_model(model, test_loader)
    print(f"Initial model size: {sum(p.numel() for p in model.parameters()) / 1e6}M")
    print(f"Initial F1 Score: {initial_metrics['f1']}")
    pruned_layers = prune_model_layers(model_path, dataset_name, f1_threshold=f1_threshold)
    print(f"Pruned Layers: {pruned_layers}")

    # Prune the model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    for layer_idx in pruned_layers:
        model.bert.encoder.layer.pop(layer_idx)

    # Update num_hidden_layers in config
    model.config.num_hidden_layers = len(model.bert.encoder.layer)

    # Save the pruned model and config
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate the pruned model
    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    eval_results = evaluate_model(model, test_loader)
    print(f"Evaluation Results after Pruning: {eval_results}")
    return eval_results


def main():
    dataset_name = "chameleon-lizard/multilingual_refusals"
    base_model = "intfloat/multilingual-e5-small"
    
    # 1. Initial Training
    initial_train_output = "initial_training"
    run_training(base_model, initial_train_output, dataset_name, epochs=1, lr=1e-4, batch_size=64, initial_train=True)

    # 2. Prune the model
    pruned_model_output = "pruned_model"
    run_pruning(initial_train_output, dataset_name, pruned_model_output)

    # 3. Train again on the pruned model
    retrained_output = "retrained_model"
    run_training(pruned_model_output, retrained_output, dataset_name, epochs=1, lr=1e-4, batch_size=64)

    # 4. Train again on the stripped model
    final_train_output = "final_training"
    run_training(stripped_model_output, final_train_output, dataset_name, epochs=1, lr=1e-4, batch_size=64)

    print("Orchestration Complete.")

if __name__ == "__main__":
    main()
