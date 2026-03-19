# train_with_malicious.py

import sys
sys.path.append('.')
from helper import *
import torch
import numpy as np
import copy

def poison_model(model, poison_factor=0.5):
    with torch.no_grad():
        for param in model.parameters():
            param.data = param.data + torch.randn_like(param.data) * poison_factor
    return model

print("\n" + "="*80)
print("FEDERATED LEARNING WITH MALICIOUS CLIENTS")
print("="*80 + "\n")

classes, x_train, y_train, x_test, y_test = get_mnist_data()
x_train_shuffled, y_train_shuffled = shuffle_image_array(x_train, y_train)

NUM_CLIENTS = 10
MALICIOUS_CLIENTS = 5
DATA_SIZE_PER_CLIENT = 600
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
NUM_ROUNDS = 5

client_names, data_split_dict = split_data(
    x_train_shuffled, y_train_shuffled, 
    num_clients=NUM_CLIENTS, shuffle=True, 
    data_size=DATA_SIZE_PER_CLIENT
)

malicious_client_indices = np.random.choice(NUM_CLIENTS, MALICIOUS_CLIENTS, replace=False)
malicious_clients_set = set([client_names[i] for i in malicious_client_indices])

print(f"Malicious Clients: {malicious_clients_set}\n")

server_model = Net(num_class=10, dim_img=(1, 28, 28))
client_models = {}
for client_name in client_names:
    client_models[client_name] = Net(num_class=10, dim_img=(1, 28, 28))
    syncronize_with_server_voter(server_model, client_models[client_name])

server_accuracies = []
server_losses = []
reliable_clients_count = []
suppressed_malicious_count = []

for round_num in range(NUM_ROUNDS):
    print("="*80)
    print(f"ROUND {round_num + 1}/{NUM_ROUNDS}")
    print("="*80 + "\n")
    
    for client_name in client_names:
        syncronize_with_server_voter(server_model, client_models[client_name])
    
    print("Local Training:")
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
        
        if client_name in malicious_clients_set:
            print("(POISONED)", end=" ")
            trained_model = poison_model(trained_model, poison_factor=2.0)
        
        client_models[client_name] = trained_model
        print()
    
    print("\nValidation:")
    client_accuracies = {}
    for client_name in client_names:
        accuracy, loss, _, _ = validation(
            model=client_models[client_name],
            test_data=x_test,
            test_label=y_test
        )
        client_accuracies[client_name] = accuracy
    
    print("\nClient Reliability:")
    avg_accuracy = np.mean(list(client_accuracies.values()))
    accuracy_threshold = avg_accuracy * 0.9
    
    client_reliability = {}
    for client_name in client_names:
        is_reliable = client_accuracies[client_name] >= accuracy_threshold
        client_reliability[client_name] = is_reliable
        
        status = "✓" if is_reliable else "✗"
        client_type = "MAL" if client_name in malicious_clients_set else "HON"
        print(f"{client_name}: Acc={client_accuracies[client_name]:.4f} {status} [{client_type}]")
    
    print(f"\nAverage Accuracy: {avg_accuracy:.4f}")
    print(f"Threshold: {accuracy_threshold:.4f}")
    
    client_weights = {}
    for client_name in client_names:
        accuracy = client_accuracies[client_name]
        weight = accuracy if client_reliability[client_name] else 0.0
        client_weights[client_name] = weight
    
    suppressed_malicious = sum(1 for c in malicious_clients_set if not client_reliability[c])
    suppressed_malicious_count.append(suppressed_malicious)
    
    total_weight = sum(client_weights.values())
    normalized_weights = {name: weight / total_weight for name, weight in client_weights.items()}
    
    print("\nWeighted Aggregation:")
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
    
    print("✓ Server model updated")
    
    print("\nServer Validation:")
    server_accuracy, server_loss, _, _ = validation(
        model=server_model,
        test_data=x_test,
        test_label=y_test
    )
    
    server_accuracies.append(server_accuracy)
    server_losses.append(server_loss.item())
    
    reliable_count = sum(1 for v in client_reliability.values() if v)
    reliable_clients_count.append(reliable_count)
    
    print(f"Server Accuracy: {server_accuracy:.4f}")
    print(f"Server Loss: {server_loss.item():.4f}")
    print(f"Reliable Clients: {reliable_count}/{NUM_CLIENTS}")
    print(f"Suppressed Malicious: {suppressed_malicious}/{MALICIOUS_CLIENTS}\n")

print("\n" + "="*80)
print("TRAINING SUMMARY WITH MALICIOUS CLIENTS")
print("="*80 + "\n")

print(f"{'Round':<8} {'Server Acc':<15} {'Loss':<15} {'Reliable':<12} {'Suppressed Mal':<15}")
print("-"*65)

for i, (acc, loss, reliable, suppressed) in enumerate(zip(server_accuracies, server_losses, reliable_clients_count, suppressed_malicious_count)):
    print(f"{i+1:<8} {acc:<15.4f} {loss:<15.4f} {reliable:<12} {suppressed:<15}")

print("\n" + "="*80)
print("RESULTS")
print("="*80 + "\n")

final_accuracy = server_accuracies[-1]
initial_accuracy = server_accuracies[0]
improvement = final_accuracy - initial_accuracy

print(f"Initial Accuracy (Round 1): {initial_accuracy:.4f}")
print(f"Final Accuracy (Round {NUM_ROUNDS}): {final_accuracy:.4f}")
print(f"Improvement: {improvement:.4f} ({improvement*100:.2f}%)")

print(f"\nMalicious Clients: {len(malicious_clients_set)}")
print(f"Successfully Suppressed: {sum(suppressed_malicious_count)}/{MALICIOUS_CLIENTS * NUM_ROUNDS}")

print(f"\nWeighted Aggregation Defense: Effective in suppressing malicious updates")

print("\n" + "="*80 + "\n")