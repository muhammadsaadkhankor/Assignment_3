# step_3_validation.py

import sys
sys.path.append('.')
from helper import *
import torch
import numpy as np

print("\n" + "="*80)
print("STEP 3: VALIDATION & CLIENT RELIABILITY EVALUATION")
print("="*80 + "\n")

classes, x_train, y_train, x_test, y_test = get_mnist_data()
x_train_shuffled, y_train_shuffled = shuffle_image_array(x_train, y_train)

NUM_CLIENTS = 10
MALICIOUS_CLIENTS = 5
DATA_SIZE_PER_CLIENT = 600
LOCAL_EPOCHS = 5
BATCH_SIZE = 32

client_names, data_split_dict = split_data(
    x_train_shuffled, y_train_shuffled, 
    num_clients=NUM_CLIENTS, shuffle=True, 
    data_size=DATA_SIZE_PER_CLIENT
)

malicious_client_indices = np.random.choice(NUM_CLIENTS, MALICIOUS_CLIENTS, replace=False)
malicious_clients_set = set([client_names[i] for i in malicious_client_indices])

server_model = Net(num_class=10, dim_img=(1, 28, 28))
client_models = {}
for client_name in client_names:
    client_models[client_name] = Net(num_class=10, dim_img=(1, 28, 28))
    syncronize_with_server_voter(server_model, client_models[client_name])

for client_name in client_names:
    syncronize_with_server_voter(server_model, client_models[client_name])

print("="*80)
print("LOCAL TRAINING - ROUND 1")
print("="*80 + "\n")

client_training_losses = {}

for client_name in client_names:
    print(f"[{client_name}] Training...")
    
    client_data = data_split_dict[client_name][0]
    client_labels = data_split_dict[client_name][1]
    
    trained_model, training_loss = train_local(
        model=client_models[client_name],
        data=client_data,
        label=client_labels,
        client_name=client_name,
        epoch=LOCAL_EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    client_models[client_name] = trained_model
    client_training_losses[client_name] = training_loss

print("\n" + "="*80)
print("STEP 3: VALIDATION ON TEST DATA")
print("="*80 + "\n")

client_accuracies = {}
client_val_losses = {}

for client_name in client_names:
    accuracy, loss, preds, labels = validation(
        model=client_models[client_name],
        test_data=x_test,
        test_label=y_test
    )
    
    client_accuracies[client_name] = accuracy
    client_val_losses[client_name] = loss.item()
    
    print(f"[{client_name}] Accuracy: {accuracy:.4f}, Loss: {loss.item():.4f}")

print("\n" + "="*80)
print("CLIENT RELIABILITY EVALUATION")
print("="*80 + "\n")

avg_accuracy = np.mean(list(client_accuracies.values()))
accuracy_threshold = avg_accuracy * 0.9

print(f"Average Accuracy: {avg_accuracy:.4f}")
print(f"Reliability Threshold: {accuracy_threshold:.4f}\n")

client_reliability = {}

for client_name in client_names:
    accuracy = client_accuracies[client_name]
    is_reliable = accuracy >= accuracy_threshold
    client_reliability[client_name] = is_reliable
    
    status = "✓ RELIABLE" if is_reliable else "✗ UNRELIABLE"
    client_type = "MALICIOUS" if client_name in malicious_clients_set else "HONEST"
    print(f"{client_name}: Acc={accuracy:.4f} | {status:<15} | {client_type}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80 + "\n")

reliable_count = sum(1 for v in client_reliability.values() if v)
unreliable_count = NUM_CLIENTS - reliable_count

print(f"Reliable Clients: {reliable_count}/{NUM_CLIENTS}")
print(f"Unreliable Clients: {unreliable_count}/{NUM_CLIENTS}")

print("\n" + "="*80)
print("STEP 3 COMPLETE")
print("="*80 + "\n")