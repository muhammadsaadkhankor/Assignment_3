# step_2_local_training.py

import sys
sys.path.append('.')
from helper import *
import torch
import numpy as np

print("\n" + "="*80)
print("STEP 2: LOCAL TRAINING ON CLIENT DEVICES")
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

print("Synchronizing server model with all clients...")
for client_name in client_names:
    syncronize_with_server_voter(server_model, client_models[client_name])
print("✓ All clients synchronized\n")

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
print("LOCAL TRAINING SUMMARY")
print("="*80 + "\n")

print(f"{'Client':<12} {'Loss':<12} {'Type':<15}")
print("-"*40)

for client_name in client_names:
    client_type = "MALICIOUS" if client_name in malicious_clients_set else "HONEST"
    loss = client_training_losses[client_name]
    print(f"{client_name:<12} {loss:<12.4f} {client_type:<15}")

print("\n" + "="*80)
print("STEP 2 COMPLETE")
print("="*80 + "\n")