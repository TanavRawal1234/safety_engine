from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import torch
import evaluate
from sklearn.model_selection import train_test_split
import json

data = pd.read_csv('main_final_data.csv')


label_columns = data.columns[1:]  
data['labels'] = data[label_columns].values.tolist()  

# Split data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)

# Load tokenizer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Preprocessing function
def preprocess_function(examples):
    tokenized_inputs = tokenizer(
        examples['comment_text'],  
        padding='max_length',
        truncation=True,
        max_length=128
    )
    
    tokenized_inputs['labels'] = examples['labels']
    return tokenized_inputs

# Tokenize datasets
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)

# Set format for PyTorch
tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_columns),  
    problem_type='multi_label_classification'
)

# Metrics
accuracy_metric = evaluate.load('accuracy')
f1_metric = evaluate.load('f1')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
    predictions = (probabilities > 0.5).astype(int)

    flat_predictions = predictions.ravel()
    flat_labels = labels.ravel().astype(int)  # Flatten for F1

    # Accuracy
    accuracy = accuracy_metric.compute(
        predictions=flat_predictions,
        references=flat_labels
    )

    # F1 score
    f1 = f1_metric.compute(
        predictions=flat_predictions,
        references=flat_labels,
        average='micro'
    )

    return {
        'accuracy': accuracy['accuracy'],
        'f1': f1['f1']
    }

# Training arguments
training_args = TrainingArguments(
    output_dir='results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    gradient_accumulation_steps=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='f1'
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics
)

# Train the model
training_result = trainer.train()

# Save training results
training_details = {
    "train_loss": training_result.training_loss,
    "train_runtime": training_result.metrics["train_runtime"],
    "train_samples_per_second": training_result.metrics["train_samples_per_second"],
    "num_train_epochs": training_result.metrics["epoch"]
}

with open('training_results_main.json', 'w') as f:
    json.dump(training_details, f, indent=4)

# Evaluate the model
evaluation_result = trainer.evaluate()
print(trainer.evaluate())

# Save evaluation results
with open('evaluation_results_main.json', 'w') as f:
    json.dump(evaluation_result, f, indent=4)

# Save the tokenizer and model
tokenizer.save_pretrained('safety_engine_model_main')
model.save_pretrained('safety_engine_model_main')
