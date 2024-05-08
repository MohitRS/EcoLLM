from datasets import load_dataset

# Load the Databricks Dolly 15k dataset
dataset = load_dataset("databricks/databricks-dolly-15k")

# Print dataset structure
print("Dataset structure:", dataset)

# Accessing the train split (adjust as necessary if the dataset has different splits)
train = dataset['train']

# Example: Print the first three samples from the training set
for i in range(3):
    print(f"Sample {i+1}: {train[i]}\n")
