import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import torch
from datasets import load_dataset
from prune import prune_model_layers


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


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=512)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=['Non-Refusal', 'Refusal'], output_dict=True)
    return {"accuracy": acc, "f1": report['Refusal']['f1-score'], "precision": report['Refusal']['precision'], "recall": report['Refusal']['recall']}

def train_classifier(
    dataset_name='chameleon-lizard/multilingual_refusals',
    model_name="xlm-roberta-base",
    output_dir="./refusal_classifier",
    batch_size=16,
    epochs=3,
    lr=2e-5,
    seed=42,
    f1_prune_threshold=0.8,
    max_length = 512
):
    """
    Trains a sequence classification model to detect refusals.

    Args:
        dataset_name (str): Name of the dataset in Hugging Face.
        model_name (str): Name of the model to use.
        output_dir (str): Directory to save the trained model.
        batch_size (int): Batch size for training and evaluation.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        seed (int): Random seed.
        prune (bool): Prune the model if True
        f1_prune_threshold (float): F1 threshold to continue pruning if pruning is true.
        max_length (int): Maximum sequence length for tokenizer
    Returns:
        dict: Evaluation results.
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load and prepare data
    data = load_and_prepare_data(dataset_name)

    # Split into train and test
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=seed, stratify=data['label'])

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize data
    train_encodings = tokenize_function(train_df.to_dict(orient='list'), tokenizer)
    test_encodings = tokenize_function(test_df.to_dict(orient='list'), tokenizer)
    
    # Convert to torch Dataset
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

    train_dataset = RefusalDataset(train_encodings, train_df['label'].tolist())
    test_dataset = RefusalDataset(test_encodings, test_df['label'].tolist())

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=seed,
    )

    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    print("Evaluation results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value}")

    # Save the model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return eval_results

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train sequence classification model to detect refusals.")
    parser.add_argument("--dataset_name", type=str, default="chameleon-lizard/multilingual_refusals", 
                        help="Name of the dataset from Hugging Face.")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base", 
                        help="Name of the model to use")
    parser.add_argument("--output_dir", type=str, default="./refusal_classifier", 
                        help="Directory to save the trained model.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument('--prune', action='store_true', help='If true, prune the model after training.')
    parser.add_argument("--f1_prune_threshold", type=float, default=0.8, help="F1 threshold for pruning.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenizer")
    args = parser.parse_args()
    
    # Train the model
    eval_results = train_classifier(
        dataset_name = args.dataset_name,
        model_name=args.model_name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        f1_prune_threshold=args.f1_prune_threshold,
        max_length=args.max_length
    )
