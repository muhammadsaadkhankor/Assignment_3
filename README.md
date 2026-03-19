# Assignment_3

Federated Learning with Weighted Aggregation — a defense mechanism against malicious clients using reliability-based weighted aggregation on the MNIST dataset.

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended)

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install PyTorch with CUDA support (adjust CUDA version as needed):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
> For CPU-only: `pip install torch torchvision`

3. Install remaining dependencies:
```bash
pip install -r requirements.txt
```

## How to Run

Run the main federated learning training with weighted aggregation:
```bash
python train.py
```

This will:
- Load and split MNIST data across 10 clients (5 honest, 5 malicious)
- Run 5 rounds of federated learning
- Apply model poisoning to malicious clients
- Use reliability-based weighted aggregation to suppress malicious updates
- Save results to `results.npy` and `client_report_data.npy`

### Other Scripts

| Script | Description |
|---|---|
| `test.py` | Data loading and client setup (Step 1) |
| `local_training.py` | Local training on client devices (Step 2) |
| `client_validation.py` | Validation and client reliability evaluation (Step 3) |
| `weighted_aggregation.py` | Weighted aggregation demo (Step 4) |
| `train_with_malicious.py` | Full training with malicious client poisoning |
| `helper.py` | Shared utilities: model, data loading, training, aggregation |
