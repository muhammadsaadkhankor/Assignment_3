# # train.py

# import sys
# sys.path.append('.')
# from helper import *
# import torch
# import numpy as np

# def poison_model(model, poison_factor=2.0):
#     with torch.no_grad():
#         for param in model.parameters():
#             param.data = param.data + torch.randn_like(param.data) * poison_factor
#     return model

# print("\n" + "="*80)
# print("FEDERATED LEARNING WITH WEIGHTED AGGREGATION")
# print("="*80 + "\n")

# classes, x_train, y_train, x_test, y_test = get_mnist_data()
# x_train_shuffled, y_train_shuffled = shuffle_image_array(x_train, y_train)

# NUM_CLIENTS = 10
# MALICIOUS_CLIENTS = 5
# DATA_SIZE_PER_CLIENT = 600
# LOCAL_EPOCHS = 5
# BATCH_SIZE = 32
# NUM_ROUNDS = 5

# client_names, data_split_dict = split_data(
#     x_train_shuffled, y_train_shuffled, 
#     num_clients=NUM_CLIENTS, shuffle=True, 
#     data_size=DATA_SIZE_PER_CLIENT
# )

# malicious_client_indices = np.random.choice(NUM_CLIENTS, MALICIOUS_CLIENTS, replace=False)
# malicious_clients_set = set([client_names[i] for i in malicious_client_indices])

# print(f"Configuration:")
# print(f"  - Total Clients: {NUM_CLIENTS}")
# print(f"  - Malicious Clients: {MALICIOUS_CLIENTS} (50%)")
# print(f"  - Honest Clients: {NUM_CLIENTS - MALICIOUS_CLIENTS} (50%)")
# print(f"  - Training Rounds: {NUM_ROUNDS}")
# print(f"  - Local Epochs: {LOCAL_EPOCHS}\n")

# server_model = Net(num_class=10, dim_img=(1, 28, 28))
# client_models = {}
# for client_name in client_names:
#     client_models[client_name] = Net(num_class=10, dim_img=(1, 28, 28))
#     syncronize_with_server_voter(server_model, client_models[client_name])

# results = {
#     'round': [],
#     'server_accuracy': [],
#     'server_loss': [],
#     'reliable_clients': [],
#     'suppressed_malicious': []
# }

# for round_num in range(NUM_ROUNDS):
#     print("="*80)
#     print(f"ROUND {round_num + 1}/{NUM_ROUNDS}")
#     print("="*80)
    
#     for client_name in client_names:
#         syncronize_with_server_voter(server_model, client_models[client_name])
    
#     for client_name in client_names:
#         client_data = data_split_dict[client_name][0]
#         client_labels = data_split_dict[client_name][1]
        
#         trained_model, _ = train_local(
#             model=client_models[client_name],
#             data=client_data,
#             label=client_labels,
#             client_name=client_name,
#             epoch=LOCAL_EPOCHS,
#             batch_size=BATCH_SIZE
#         )
        
#         if client_name in malicious_clients_set:
#             trained_model = poison_model(trained_model, poison_factor=2.0)
        
#         client_models[client_name] = trained_model
    
#     client_accuracies = {}
#     for client_name in client_names:
#         accuracy, _, _, _ = validation(
#             model=client_models[client_name],
#             test_data=x_test,
#             test_label=y_test
#         )
#         client_accuracies[client_name] = accuracy
    
#     avg_accuracy = np.mean(list(client_accuracies.values()))
#     accuracy_threshold = avg_accuracy * 0.9
    
#     client_reliability = {}
#     for client_name in client_names:
#         client_reliability[client_name] = client_accuracies[client_name] >= accuracy_threshold
    
#     client_weights = {}
#     for client_name in client_names:
#         accuracy = client_accuracies[client_name]
#         weight = accuracy if client_reliability[client_name] else 0.0
#         client_weights[client_name] = weight
    
#     suppressed_malicious = sum(1 for c in malicious_clients_set if not client_reliability[c])
    
#     total_weight = sum(client_weights.values())
#     normalized_weights = {name: weight / total_weight for name, weight in client_weights.items()}
    
#     if torch.cuda.is_available():
#         server_model = server_model.cuda()
#     server_params = {name: param.data.clone() for name, param in server_model.named_parameters()}
    
#     with torch.no_grad():
#         aggregated_delta = {
#             name: torch.zeros_like(param.data)
#             for name, param in server_model.named_parameters()
#         }
        
#         for client_name in client_names:
#             weight = normalized_weights[client_name]
            
#             for name, param in client_models[client_name].named_parameters():
#                 delta = param.data - server_params[name]
#                 aggregated_delta[name] += weight * delta
        
#         for name, param in server_model.named_parameters():
#             param.data = server_params[name] + aggregated_delta[name]
    
#     server_accuracy, server_loss, _, _ = validation(
#         model=server_model,
#         test_data=x_test,
#         test_label=y_test
#     )
    
#     reliable_count = sum(1 for v in client_reliability.values() if v)
    
#     results['round'].append(round_num + 1)
#     results['server_accuracy'].append(server_accuracy)
#     results['server_loss'].append(server_loss.item())
#     results['reliable_clients'].append(reliable_count)
#     results['suppressed_malicious'].append(suppressed_malicious)
    
#     print(f"\nRound {round_num + 1} Results:")
#     print(f"  - Server Accuracy: {server_accuracy:.4f}")
#     print(f"  - Server Loss: {server_loss.item():.4f}")
#     print(f"  - Reliable Clients: {reliable_count}/{NUM_CLIENTS}")
#     print(f"  - Suppressed Malicious: {suppressed_malicious}/{MALICIOUS_CLIENTS}\n")

# print("\n" + "="*80)
# print("FINAL RESULTS")
# print("="*80 + "\n")

# print(f"{'Round':<8} {'Accuracy':<15} {'Loss':<15} {'Reliable':<12} {'Suppressed':<12}")
# print("-"*62)

# for i in range(NUM_ROUNDS):
#     print(f"{results['round'][i]:<8} {results['server_accuracy'][i]:<15.4f} {results['server_loss'][i]:<15.4f} {results['reliable_clients'][i]:<12} {results['suppressed_malicious'][i]:<12}")

# print("\n" + "="*80)

# initial_acc = results['server_accuracy'][0]
# final_acc = results['server_accuracy'][-1]
# improvement = final_acc - initial_acc

# print(f"\nInitial Accuracy: {initial_acc:.4f}")
# print(f"Final Accuracy: {final_acc:.4f}")
# print(f"Improvement: {improvement:.4f} ({improvement*100:.2f}%)")

# print(f"\nTotal Malicious Clients Suppressed: {sum(results['suppressed_malicious'])}/{MALICIOUS_CLIENTS * NUM_ROUNDS}")
# print(f"Suppression Rate: {(sum(results['suppressed_malicious']) / (MALICIOUS_CLIENTS * NUM_ROUNDS)) * 100:.2f}%")

# print("\n" + "="*80 + "\n")

# np.save('results.npy', results)
# print("✓ Results saved to results.npy\n")

# train.py

import sys
sys.path.append('.')
from helper import *
import torch
import numpy as np

def poison_model(model, poison_factor=2.0):
    with torch.no_grad():
        for param in model.parameters():
            param.data = param.data + torch.randn_like(param.data) * poison_factor
    return model

print("\n" + "="*80)
print("FEDERATED LEARNING WITH WEIGHTED AGGREGATION")
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

print(f"Configuration:")
print(f"  - Total Clients: {NUM_CLIENTS}")
print(f"  - Malicious Clients: {MALICIOUS_CLIENTS} (50%)")
print(f"  - Honest Clients: {NUM_CLIENTS - MALICIOUS_CLIENTS} (50%)")
print(f"  - Training Rounds: {NUM_ROUNDS}")
print(f"  - Local Epochs: {LOCAL_EPOCHS}\n")

server_model = Net(num_class=10, dim_img=(1, 28, 28))
client_models = {}
for client_name in client_names:
    client_models[client_name] = Net(num_class=10, dim_img=(1, 28, 28))
    syncronize_with_server_voter(server_model, client_models[client_name])

results = {
    'round': [],
    'server_accuracy': [],
    'server_loss': [],
    'reliable_clients': [],
    'suppressed_malicious': []
}

for round_num in range(NUM_ROUNDS):
    print("="*80)
    print(f"ROUND {round_num + 1}/{NUM_ROUNDS}")
    print("="*80)
    
    for client_name in client_names:
        syncronize_with_server_voter(server_model, client_models[client_name])
    
    for client_name in client_names:
        client_data = data_split_dict[client_name][0]
        client_labels = data_split_dict[client_name][1]
        
        trained_model, _ = train_local(
            model=client_models[client_name],
            data=client_data,
            label=client_labels,
            client_name=client_name,
            epoch=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE
        )
        
        if client_name in malicious_clients_set:
            trained_model = poison_model(trained_model, poison_factor=2.0)
        
        client_models[client_name] = trained_model
    
    client_accuracies = {}
    for client_name in client_names:
        accuracy, _, _, _ = validation(
            model=client_models[client_name],
            test_data=x_test,
            test_label=y_test
        )
        client_accuracies[client_name] = accuracy
    
    avg_accuracy = np.mean(list(client_accuracies.values()))
    accuracy_threshold = avg_accuracy * 0.9
    
    client_reliability = {}
    for client_name in client_names:
        client_reliability[client_name] = client_accuracies[client_name] >= accuracy_threshold
    
    client_weights = {}
    for client_name in client_names:
        accuracy = client_accuracies[client_name]
        weight = accuracy if client_reliability[client_name] else 0.0
        client_weights[client_name] = weight
    
    suppressed_malicious = sum(1 for c in malicious_clients_set if not client_reliability[c])
    
    total_weight = sum(client_weights.values())
    normalized_weights = {name: weight / total_weight for name, weight in client_weights.items()}
    
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
    
    server_accuracy, server_loss, _, _ = validation(
        model=server_model,
        test_data=x_test,
        test_label=y_test
    )
    
    reliable_count = sum(1 for v in client_reliability.values() if v)
    
    results['round'].append(round_num + 1)
    results['server_accuracy'].append(server_accuracy)
    results['server_loss'].append(server_loss.item())
    results['reliable_clients'].append(reliable_count)
    results['suppressed_malicious'].append(suppressed_malicious)
    
    print(f"\nRound {round_num + 1} Results:")
    print(f"  - Server Accuracy: {server_accuracy:.4f}")
    print(f"  - Server Loss: {server_loss.item():.4f}")
    print(f"  - Reliable Clients: {reliable_count}/{NUM_CLIENTS}")
    print(f"  - Suppressed Malicious: {suppressed_malicious}/{MALICIOUS_CLIENTS}\n")
    
    # Store final round data for report
    if round_num == NUM_ROUNDS - 1:
        final_client_accuracies = client_accuracies
        final_normalized_weights = normalized_weights
        final_avg_accuracy = avg_accuracy
        final_accuracy_threshold = accuracy_threshold
        final_client_reliability = client_reliability

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80 + "\n")

print(f"{'Round':<8} {'Accuracy':<15} {'Loss':<15} {'Reliable':<12} {'Suppressed':<12}")
print("-"*62)

for i in range(NUM_ROUNDS):
    print(f"{results['round'][i]:<8} {results['server_accuracy'][i]:<15.4f} {results['server_loss'][i]:<15.4f} {results['reliable_clients'][i]:<12} {results['suppressed_malicious'][i]:<12}")

print("\n" + "="*80)

initial_acc = results['server_accuracy'][0]
final_acc = results['server_accuracy'][-1]
improvement = final_acc - initial_acc

print(f"\nInitial Accuracy: {initial_acc:.4f}")
print(f"Final Accuracy: {final_acc:.4f}")
print(f"Improvement: {improvement:.4f} ({improvement*100:.2f}%)")

print(f"\nTotal Malicious Clients Suppressed: {sum(results['suppressed_malicious'])}/{MALICIOUS_CLIENTS * NUM_ROUNDS}")
print(f"Suppression Rate: {(sum(results['suppressed_malicious']) / (MALICIOUS_CLIENTS * NUM_ROUNDS)) * 100:.2f}%")

# =====================================================
# CAPTURE FINAL ROUND CLIENT-LEVEL DATA FOR REPORT
# =====================================================

print("\n" + "="*80)
print("FINAL ROUND - CLIENT-LEVEL ANALYSIS")
print("="*80 + "\n")

print(f"Average Accuracy: {final_avg_accuracy:.4f}")
print(f"Reliability Threshold (90% of avg): {final_accuracy_threshold:.4f}\n")

print(f"{'Client':<12} {'Accuracy':<15} {'Weight':<15} {'Status':<20}")
print("-"*65)

for client_name in client_names:
    accuracy = final_client_accuracies[client_name]
    weight = final_normalized_weights[client_name]
    client_type = "Malicious" if client_name in malicious_clients_set else "Honest"
    reliability = "Reliable" if final_client_reliability[client_name] else "Unreliable"
    status = f"{reliability} ({client_type})"
    print(f"{client_name:<12} {accuracy:<15.4f} {weight:<15.4f} {status:<20}")

print("\n" + "="*80 + "\n")

# Save all data
np.save('results.npy', results)

client_report_data = {
    'client_names': client_names,
    'malicious_clients': list(malicious_clients_set),
    'final_client_accuracies': final_client_accuracies,
    'final_normalized_weights': final_normalized_weights,
    'average_accuracy': final_avg_accuracy,
    'accuracy_threshold': final_accuracy_threshold
}

np.save('client_report_data.npy', client_report_data)

print("✓ Results saved to results.npy")
print("✓ Client-level data saved to client_report_data.npy\n")