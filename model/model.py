# Install required packages in Colab
# Install required packages in Colab
#!pip install transformers
#!pip install datasets
#!pip install scikit-learn
#!pip install pandas
#!pip install openpyxl

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainingArguments
)
from sklearn.model_selection import train_test_split

# Upload your Excel file in Colab before running this cell
from google.colab import files
uploaded = files.upload()  # Select your Comp-simp.xlsx file here

# Clear GPU cache (just in case)
torch.cuda.empty_cache()

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load dataset
print("Loading dataset: Comp-simp.xlsx")
df = pd.read_excel("Comp-simp.xlsx")
df = df[['complex', 'simple']].dropna()
df = df.rename(columns={"complex": "input_text", "simple": "target_text"})

# Split data
print("Splitting into training and validation data")
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Load tokenizer and model (use MT5 tokenizer for MT5 model)
model_name = "csebuetnlp/mT5_multilingual_XLSum"
print("Loading Tokenizer and Model:", model_name)
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name).to(device)

# Preprocessing
print("Preparing Dataset")
max_input_length = 128
max_target_length = 128

def preprocess_function(examples):
    inputs = [f"simplify: {text}" for text in examples["input_text"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")

    # Use text_target for label tokenization as per latest transformers versions
    labels = tokenizer(
        examples["target_text"],
        max_length=max_target_length,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_tokenized = train_dataset.map(preprocess_function, batched=True)
val_tokenized = val_dataset.map(preprocess_function, batched=True)

# Data collator for seq2seq
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Training arguments
print("Setting Training Arguments")
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Note: 'eval_strategy' changed to 'evaluation_strategy'
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    logging_steps=10
)

# Trainer setup
print("Trainer Setup")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train
print("Training the Model")
trainer.train()

# Generate predictions on validation samples
print("\nGenerating Predictions on Validation Data:")
for i in range(5):
    input_text = val_df.iloc[i]['input_text']
    target_text = val_df.iloc[i]['target_text']

    input_ids = tokenizer.encode(f"simplify: {input_text}", return_tensors="pt", max_length=128, truncation=True).to(device)
    output_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    predicted_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"\nOriginal: {input_text}")
    print(f"Predicted Simplification: {predicted_text}")
    print(f"Actual Simplification: {target_text}")

# Save the trained model and tokenizer
model_save_path = "./trained_model"
print(f"Saving model to {model_save_path}")
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
