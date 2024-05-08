from datasets import load_dataset

# Load the Wikitext-103 dataset
dataset = load_dataset("wikitext", "wikitext-103-v1")

# Accessing the train and validation splits
train = dataset['train']
validation = dataset['validation']

for i in range(10):
	print(f"sample: {i+1}: {train[i]['text']}/n")
