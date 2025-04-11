import sys
import os
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers import SentenceTransformerTrainer
from datasets import IterableDataset, Dataset, Features, Value
from datetime import datetime

# Set paths
build_project_path = r'C:\fine-tuning-build-project\Gemini2.5pro\fine-tuning-build-project'
build_project_path = r'C:\fine-tuning-build-project\Gemini2.5pro\fine-tuning-build-project'
fine_tuning_path = os.path.join(build_project_path, 'fine_tuning')

full_jitter_titles_path = os.path.join(build_project_path, 'synthetic_data', 'data', 'jittered_titles.csv')

dataset_dir_path = os.path.join(fine_tuning_path, 'data', 'datasets')
if not os.path.isdir(dataset_dir_path):
    os.makedirs(dataset_dir_path)

all_dataset_paths = [
    os.path.join(dataset_dir_path, 'train_ds.csv'),
    os.path.join(dataset_dir_path, 'val_ds.csv'),
    os.path.join(dataset_dir_path, 'test_ds.csv')
]

# Train/val/test split on seed titles (stratified by ONET code)
if all([os.path.isfile(p) for p in all_dataset_paths]):
    train_df = pd.read_csv(all_dataset_paths[0])
    val_df = pd.read_csv(all_dataset_paths[1])
    test_df = pd.read_csv(all_dataset_paths[2])
else:
    jittered_titles_df = pd.read_csv(full_jitter_titles_path)

    unique_onets = jittered_titles_df['onet_code'].unique()

    # Proportion of ONET coverage in each split
    val_p = 0.6
    test_p = 1.0

    val_mask = np.random.uniform(size=len(unique_onets)) < val_p
    test_mask = np.random.uniform(size=len(unique_onets)) < test_p

    val_clean_titles = []
    test_clean_titles = []

    for i, onet in enumerate(unique_onets):
        onet_clean_titles = jittered_titles_df[jittered_titles_df['onet_code'] == onet]['seed_title'].unique()
        if len(onet_clean_titles) < 4:
            print(f'onet {onet} has {len(onet_clean_titles)} seed titles')
        else:
            to_split_off = np.random.choice(onet_clean_titles, size=2, replace=False)
            if val_mask[i]:
                val_clean_titles.append(to_split_off[0])
            if test_mask[i]:
                test_clean_titles.append(to_split_off[1])

    val_df = jittered_titles_df[jittered_titles_df['seed_title'].isin(val_clean_titles)]
    test_df = jittered_titles_df[jittered_titles_df['seed_title'].isin(test_clean_titles)]
    train_df = jittered_titles_df[(~jittered_titles_df['seed_title'].isin(test_clean_titles)) & (~jittered_titles_df['seed_title'].isin(val_clean_titles))]

    train_df.to_csv(all_dataset_paths[0], index=False)
    val_df.to_csv(all_dataset_paths[1], index=False)
    test_df.to_csv(all_dataset_paths[2], index=False)

print(f'Train size: {len(train_df)}')
print(f'Val size: {len(val_df)}')
print(f'Test size: {len(test_df)}')

# Handle abbreviations
def split_title(job_title):
    m = re.search(r'\(.*\)$', job_title)
    if not m:
        return None
    abbreviated_title = m.group()[1:-1]
    un_abbreviated_title = job_title[:m.span()[0] - 1]
    return abbreviated_title, un_abbreviated_title

def get_clean_title(title):
    split_title_result = split_title(title)
    if split_title_result:
        return split_title_result[np.argmax([len(t) for t in split_title_result])]
    return title

# Use all-mpnet-base-v2 instead of all-MiniLM-L6-v2
base_model_name = 'sentence-transformers/all-mpnet-base-v2'

base_model = SentenceTransformer(base_model_name)

# Set up training datasets
def generate_training_examples(ds, take_longest_variant=True, infinite=True):
    rng = np.random.default_rng()

    if take_longest_variant:
        ds['seed_title'] = ds['seed_title'].apply(get_clean_title)

    ds = ds.reset_index(drop=True)
    anchors = ds['jittered_title'].to_numpy()
    positives = ds['seed_title'].to_numpy()
    seed_titles = ds['seed_title'].unique()

    negative_indices_list = []
    for positive in positives:
        negative_indices_list.append(
            np.arange(len(seed_titles))[seed_titles != positive]
        )

    indices = list(range(len(anchors)))
    while infinite:
        rng.shuffle(indices)

        for idx in indices:
            negative_indices = negative_indices_list[idx]
            negative_idx = rng.choice(negative_indices)
            yield {
                'anchor': anchors[idx],
                'positive': positives[idx],
                'negative': seed_titles[negative_idx]
            }

# Create training dataset
train_ds = IterableDataset.from_generator(
    generate_training_examples,
    gen_kwargs={'ds': train_df, 'take_longest_variant': True, 'infinite': True},
    features=Features(
        {
            'anchor': Value('string'),
            'positive': Value('string'),
            'negative': Value('string'),
        }
    ),
).with_format(None)

# Create validation dataset
def generate_val_examples(ds, take_longest_variant=True, negatives_per_positive=5):
    rng = np.random.default_rng()

    if take_longest_variant:
        ds['seed_title'] = ds['seed_title'].apply(get_clean_title)

    ds = ds.reset_index(drop=True)
    anchors = ds['jittered_title'].to_numpy()
    positives = ds['seed_title'].to_numpy()
    seed_titles = ds['seed_title'].unique()

    examples = []
    for anchor, positive in zip(anchors, positives):
        negatives = rng.choice(seed_titles[seed_titles != positive], size=negatives_per_positive, replace=False)
        for negative in negatives:
            examples.append({
                'anchor': anchor,
                'positive': positive,
                'negative': negative
            })

    return examples

val_ds = Dataset.from_list(generate_val_examples(val_df, take_longest_variant=True, negatives_per_positive=5)).with_format(None)

# Set up loss function
train_loss = losses.TripletLoss(model=base_model, triplet_margin=0.3)

# Training parameters
num_epochs = 5
train_batch_size = 192
val_batch_size = 192
evals_per_epoch = 4

# Set up model save path
model_save_path = os.path.join(fine_tuning_path, 'data', 'trained_models')
model_name = f'{base_model_name.replace("/", "-")}_triplet_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

# Configure training arguments
steps_per_epoch = len(train_df) // train_batch_size
steps_per_eval = steps_per_epoch // evals_per_epoch

train_args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=model_save_path,
    # Optional training parameters:
    max_steps=steps_per_epoch * num_epochs,
    num_train_epochs=num_epochs,
    warmup_ratio=0.1,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=val_batch_size,
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=steps_per_eval,
    save_strategy="steps",
    save_steps=steps_per_eval,
    save_total_limit=2,
    logging_steps=steps_per_eval,
    report_to='tensorboard',
    run_name=model_name,  # Will be used in W&B if `wandb` is installed
)

# Add this code after the model_save_path definition:
streamlit_model_path = os.path.join(build_project_path, 'streamlit_app', 'data', 'fine_tuned_model')
if not os.path.isdir(streamlit_model_path):
    os.makedirs(streamlit_model_path)

# Initialize and run the trainer
if __name__ == "__main__":
    trainer = SentenceTransformerTrainer(
        model=base_model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=train_args,
        loss=train_loss,
    )
    
    trainer.train()
    
    # Save model to streamlit app location
    print("Saving model for streamlit app...")
    final_model_path = os.path.join(model_save_path, model_name)
    streamlit_model_path = os.path.join(build_project_path, 'streamlit_app', 'data', 'fine_tuned_model')
    if not os.path.isdir(streamlit_model_path):
        os.makedirs(streamlit_model_path)
    
    # Copy or save model directly to streamlit location
    base_model.save(streamlit_model_path)
    print(f"Model saved to {streamlit_model_path}") 