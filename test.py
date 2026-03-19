# step_1_data_setup.py
# STEP 1: Load MNIST data, split to clients, initialize model

import sys
sys.path.append('.')
from helper import *
import torch

# ============================================================================
# STEP 1: LOAD MNIST DATA AND UNDERSTAND IT
# ============================================================================

print("\n" + "="*80)
print("STEP 1: DATA LOADING AND SPLITTING")
print("="*80 + "\n")

# Load MNIST dataset
classes, x_train, y_train, x_test, y_test = get_mnist_data()

print(f"✓ MNIST Data Loaded Successfully!")
print(f"  - Total training samples: {len(x_train)}")
print(f"  - Total test samples: {len(x_test)}")
print(f"  - Image shape: {x_train[0].shape}")
print(f"  - Classes: {len(classes)} (0-9 digits)")
print(f"  - Class names: {classes}\n")

# ============================================================================
# STEP 1B: SHUFFLE AND SPLIT DATA TO CLIENTS
# ============================================================================

# Configuration
NUM_CLIENTS = 10
MALICIOUS_CLIENTS = 5  # 50% are malicious
DATA_SIZE_PER_CLIENT = 600

# Shuffle training data
x_train_shuffled, y_train_shuffled = shuffle_image_array(x_train, y_train)

# Split data to clients
client_names, data_split_dict = split_data(
    x_train_shuffled, 
    y_train_shuffled, 
    num_clients=NUM_CLIENTS,
    shuffle=True,
    data_size=DATA_SIZE_PER_CLIENT
)

print(f"✓ Data Split to {NUM_CLIENTS} Clients!")
for client_name in client_names:
    data_size = len(data_split_dict[client_name][0])
    print(f"  - {client_name}: {data_size} samples")

# ============================================================================
# STEP 1C: IDENTIFY MALICIOUS CLIENTS
# ============================================================================

malicious_client_indices = np.random.choice(
    NUM_CLIENTS, 
    MALICIOUS_CLIENTS, 
    replace=False
)
malicious_clients_set = set([client_names[i] for i in malicious_client_indices])

print(f"\n✓ Malicious Clients Identified (50%): {malicious_clients_set}\n")

# ============================================================================
# STEP 1D: INITIALIZE SERVER AND CLIENT MODELS
# ============================================================================

# Server model (global model)
server_model = Net(num_class=10, dim_img=(1, 28, 28))

# Initialize client models (all start with server's weights)
client_models = {}
for client_name in client_names:
    client_models[client_name] = Net(num_class=10, dim_img=(1, 28, 28))
    # Synchronize client with server (same initial weights)
    syncronize_with_server_voter(server_model, client_models[client_name])

print(f"✓ Server Model & Client Models Initialized!")
print(f"  - All {NUM_CLIENTS} clients have same initial weights as server\n")

# ============================================================================
# STEP 1E: PREPARE TEST DATA FOR VALIDATION
# ============================================================================

print(f"✓ Test Data Ready for Validation!")
print(f"  - Test samples: {len(x_test)}")
print(f"  - Test data shape: {x_test[0].shape}\n")

# ============================================================================
# SUMMARY OF STEP 1
# ============================================================================

print("="*80)
print("STEP 1 SUMMARY")
print("="*80)
print(f"✓ Total Clients: {NUM_CLIENTS}")
print(f"✓ Honest Clients: {NUM_CLIENTS - MALICIOUS_CLIENTS}")
print(f"✓ Malicious Clients: {MALICIOUS_CLIENTS} (50%)")
print(f"✓ Data per Client: {DATA_SIZE_PER_CLIENT} samples")
print(f"✓ Global Model: Initialized & Ready")
print(f"✓ Client Models: Initialized & Synced with Server")
print(f"✓ Test Data: Ready for validation")
print("="*80 + "\n")