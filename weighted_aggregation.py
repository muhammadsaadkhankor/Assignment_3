# step_4_weighted_aggregation.py

import sys
sys.path.append('.')
from helper import *
import torch
import numpy as np
import copy

print("\n" + "="*80)
print("STEP 4: WEIGHTED AGGREGATION")
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
    print(f"[{client_name}] Training...", end=" ")
    
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
print("VALIDATION ON TEST DATA")
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

avg_accuracy = np.mean(list(client_accuracies.values()))
accuracy_threshold = avg_accuracy * 0.9

client_reliability = {}
for client_name in client_names:
    client_reliability[client_name] = client_accuracies[client_name] >= accuracy_threshold

print("\n" + "="*80)
print("ASSIGNING WEIGHTS TO CLIENTS")
print("="*80 + "\n")

client_weights = {}

for client_name in client_names:
    accuracy = client_accuracies[client_name]
    
    if client_reliability[client_name]:
        weight = accuracy
    else:
        weight = 0.0
    
    client_weights[client_name] = weight
    
    status = "✓" if client_reliability[client_name] else "✗"
    print(f"{client_name}: Accuracy={accuracy:.4f}, Weight={weight:.4f} {status}")

total_weight = sum(client_weights.values())
normalized_weights = {name: weight / total_weight for name, weight in client_weights.items()}

print(f"\nTotal Weight: {total_weight:.4f}")
print(f"\nNormalized Weights:")
for client_name in client_names:
    print(f"{client_name}: {normalized_weights[client_name]:.4f}")

print("\n" + "="*80)
print("WEIGHTED AGGREGATION")
print("="*80 + "\n")

if torch.cuda.is_available():
    server_model = server_model.cuda()

server_params = {name: param.data.clone() for name, param in server_model.named_parameters()}

with torch.no_grad():
    aggregated_delta = {
        name: torch.zeros_like(param.data)
        for name, param in server_model.named_parameters()
    }
    
    for client_name in client_names:
        weight = normalized_weights[client_name]
        
        for name, param in client_models[client_name].named_parameters():
            delta = param.data - server_params[name]
            aggregated_delta[name] += weight * delta
    
    for name, param in server_model.named_parameters():
        param.data = server_params[name] + aggregated_delta[name]

print("✓ Server model updated with weighted aggregation")

print("\n" + "="*80)
print("SERVER MODEL VALIDATION")
print("="*80 + "\n")

server_accuracy, server_loss, _, _ = validation(
    model=server_model,
    test_data=x_test,
    test_label=y_test
)

print(f"Server Model Accuracy: {server_accuracy:.4f}")
print(f"Server Model Loss: {server_loss.item():.4f}")

print("\n" + "="*80)
print("STEP 4 COMPLETE")
print("="*80 + "\n")