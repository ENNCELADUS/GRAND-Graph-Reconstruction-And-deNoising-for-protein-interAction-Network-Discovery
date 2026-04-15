# Model Architecture Standards

This document defines the architectural standards for implementing models in the GRAND system. The goal is to maintain a modular, predictable, and clean codebase where new architectures can be added without modifying the core orchestration logic.

## Design Philosophy

1.  **Isolation**: Each model architecture resides in its own independent file within `src/model/`.
2.  **Standard Contract**: Models must adhere to the standard PyTorch `nn.Module` interface. No custom base classes or complex inheritance hierarchies are enforced.
3.  **Config Injection**: Models receive only the configuration parameters relevant to them, filtered by the engine.

## Implementation Standards

### 1. File Structure
*   **Path**: `src/model/{model_name}.py`
*   **Content**: A single class inheriting from `nn.Module` (e.g., `class V3(nn.Module): ...`).
*   **Helper Modules**: Architecture-specific sub-modules (blocks, layers) should be defined within the same file or a private utility, keeping the public namespace clean.

### 2. The Contract (`nn.Module`)

Models must implement:

*   **`__init__(self, **kwargs)`**:
    *   Accepts configuration parameters as keyword arguments.
    *   Initializes all layers and sub-modules.
*   **`forward(self, **batch)`**:
    *   Accepts input data as named arguments matching the data loader output keys (`emb_a`, `emb_b`, `len_a`, `len_b`, `label`).
    *   Returns a dictionary containing at least `logits`. When `label` is present in the input, also returns `loss`.

### 3. Configuration Handling

*   **Extraction**: The engine (`src/pipeline/engine.py`) uses `src/utils/config.py` to extract model-specific parameters from the global config via `extract_model_kwargs()`.
*   **Injection**: These parameters are passed to the model's `__init__`.
*   **Strictness**: Models should not be aware of the global config structure. They only know about their specific hyperparameters (e.g., `d_model`, `n_layers`, `dropout`).

## Example Implementation

```python
# src/model/v3.py
import torch.nn as nn

class V3(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, dropout=0.1):
        super().__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead=8, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, 1)

    def forward(self, emb_a, emb_b, len_a, len_b, label=None, **kwargs):
        # ... encoding and cross-attention logic ...
        logits = self.head(cls_repr)

        output = {"logits": logits}
        if label is not None:
            output["loss"] = nn.functional.binary_cross_entropy_with_logits(logits, label)

        return output
```

## Instantiation Logic

The engine (`src/pipeline/engine.py`) handles model selection via `build_model()` in `src/pipeline/stages/train.py`, using a `MODEL_FACTORIES` dispatch dict:

```python
MODEL_FACTORIES: dict[str, ModelFactory] = {
    "v3":   _build_v3_model,
    "v3.1": _build_v3_1_model,
    "v4":   _build_v4_model,
    "v5":   _build_v5_model,
}

def build_model(config: ConfigDict) -> nn.Module:
    model_name, model_kwargs = extract_model_kwargs(config)
    factory = MODEL_FACTORIES.get(model_name)
    if factory is not None:
        return factory(model_kwargs)
    raise ValueError(f"Unknown model: {model_name}")
```

After instantiation, the engine moves the model to the runtime device (`model.to(runtime.device)`) before passing it to any stage.
