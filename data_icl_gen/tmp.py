def batch_list(input_list, batch_size):
    """Cut a list into batches according to the specified batch size."""
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]

# Example usage
input_list = list(range(1, 23))  # Example list
batch_size = 999
batches = batch_list(input_list, batch_size)

for i, batch in enumerate(batches):
    print(f"Batch {i + 1}: {batch}")