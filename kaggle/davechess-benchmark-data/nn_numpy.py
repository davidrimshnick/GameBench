"""Pure-numpy inference engine for the DaveChess AlphaZero network.

Zero torch dependency at inference time.  Torch is only used for the
one-time weight conversion from .pt checkpoint to .npz format.

Usage:
    # Convert checkpoint once:
    python nn_numpy.py

    # Then load and run inference (no torch needed):
    from nn_numpy import NumpyNetwork
    net = NumpyNetwork.from_npz("model_weights.npz")
    policy_logits, value = net.predict(planes)  # planes: (18, 8, 8)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────
BOARD_SIZE = 8
MOVES_PER_SQUARE = 67
POLICY_SIZE = BOARD_SIZE * BOARD_SIZE * MOVES_PER_SQUARE  # 4288

ALL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
DIR_TO_IDX = {d: i for i, d in enumerate(ALL_DIRS)}

BN_EPS = 1e-5  # PyTorch default


# ── Numpy primitives ────────────────────────────────────────────────────

def conv2d(x, w, padding=0):
    """2-D convolution (NCHW) via im2col.

    Args:
        x: (N, C_in, H, W)
        w: (C_out, C_in, kH, kW)
        padding: int, symmetric zero-padding

    Returns:
        (N, C_out, H_out, W_out)
    """
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    N, C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape
    H_out = H - kH + 1
    W_out = W - kW + 1

    # im2col: build (N, C_in*kH*kW, H_out*W_out) matrix
    col = _im2col(x, kH, kW, H_out, W_out)
    # w reshaped to (C_out, C_in*kH*kW)
    w_col = w.reshape(C_out, -1)
    # matmul: for each sample, (C_out, C_in*kH*kW) @ (C_in*kH*kW, H_out*W_out)
    # col is (N, C_in*kH*kW, H_out*W_out)
    # Use einsum: 'oi,nik->nok' contracts over i (the C_in*kH*kW axis)
    out = np.einsum('oi,nik->nok', w_col, col, optimize=True)
    return out.reshape(N, C_out, H_out, W_out)


def _im2col(x, kH, kW, H_out, W_out):
    """Extract patches as columns for convolution.

    Args:
        x: (N, C, H, W)  -- already padded
    Returns:
        (N, C*kH*kW, H_out*W_out)
    """
    N, C, H, W = x.shape
    # Use stride tricks for zero-copy view
    s = x.strides
    shape = (N, C, kH, kW, H_out, W_out)
    strides = (s[0], s[1], s[2], s[3], s[2], s[3])
    patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    # (N, C, kH, kW, H_out, W_out) -> (N, C*kH*kW, H_out*W_out)
    return patches.reshape(N, C * kH * kW, H_out * W_out)


def batch_norm(x, gamma, beta, running_mean, running_var):
    """Batch normalization in eval mode (uses running stats).

    Args:
        x: (N, C, H, W)
        gamma, beta, running_mean, running_var: (C,)

    Returns:
        (N, C, H, W)
    """
    # Reshape for broadcasting: (1, C, 1, 1)
    mean = running_mean.reshape(1, -1, 1, 1)
    var = running_var.reshape(1, -1, 1, 1)
    g = gamma.reshape(1, -1, 1, 1)
    b = beta.reshape(1, -1, 1, 1)
    return g * (x - mean) / np.sqrt(var + BN_EPS) + b


def relu(x):
    """ReLU activation."""
    return np.maximum(x, 0)


def linear(x, w, b):
    """Fully-connected layer: x @ w^T + b.

    Args:
        x: (N, in_features)
        w: (out_features, in_features)
        b: (out_features,)

    Returns:
        (N, out_features)
    """
    return x @ w.T + b


# ── Network class ────────────────────────────────────────────────────────

class NumpyNetwork:
    """Pure-numpy DaveChess AlphaZero network.

    Replicates the PyTorch DaveChessNetwork forward pass exactly,
    using running batch-norm statistics (eval mode).
    """

    def __init__(self, weights: dict[str, np.ndarray]):
        """Initialize from a flat dict of weight name -> numpy array.

        The key names match the PyTorch state_dict convention.
        """
        self.w = weights

        # Infer architecture
        self.num_filters = self.w["conv_input.weight"].shape[0]
        self.input_planes = self.w["conv_input.weight"].shape[1]
        self.num_res_blocks = max(
            int(k.split(".")[1])
            for k in self.w if k.startswith("res_blocks.")
        ) + 1

    def predict(self, planes: np.ndarray) -> tuple[np.ndarray, float]:
        """Run inference on a single position.

        Args:
            planes: (18, 8, 8) float32 input planes from state_to_planes()

        Returns:
            (policy_logits, value) where policy_logits is (4288,) float32
            and value is a float in [-1, 1].
        """
        x = planes[np.newaxis].astype(np.float32)  # (1, 18, 8, 8)
        return self._forward(x)

    def predict_batch(self, planes_batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on a batch of positions.

        Args:
            planes_batch: (N, 18, 8, 8) float32

        Returns:
            (policy_logits, values) where policy_logits is (N, 4288)
            and values is (N,) floats in [-1, 1].
        """
        x = planes_batch.astype(np.float32)
        logits, values = self._forward(x)
        return logits, values

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, float | np.ndarray]:
        """Full forward pass.

        Args:
            x: (N, C_in, 8, 8) float32

        Returns:
            (policy_logits, value)
        """
        w = self.w

        # ── Common trunk ──
        x = conv2d(x, w["conv_input.weight"], padding=1)
        x = batch_norm(x, w["bn_input.weight"], w["bn_input.bias"],
                       w["bn_input.running_mean"], w["bn_input.running_var"])
        x = relu(x)

        for i in range(self.num_res_blocks):
            residual = x
            # conv1 + bn1 + relu
            x = conv2d(x, w[f"res_blocks.{i}.conv1.weight"], padding=1)
            x = batch_norm(x, w[f"res_blocks.{i}.bn1.weight"],
                           w[f"res_blocks.{i}.bn1.bias"],
                           w[f"res_blocks.{i}.bn1.running_mean"],
                           w[f"res_blocks.{i}.bn1.running_var"])
            x = relu(x)
            # conv2 + bn2
            x = conv2d(x, w[f"res_blocks.{i}.conv2.weight"], padding=1)
            x = batch_norm(x, w[f"res_blocks.{i}.bn2.weight"],
                           w[f"res_blocks.{i}.bn2.bias"],
                           w[f"res_blocks.{i}.bn2.running_mean"],
                           w[f"res_blocks.{i}.bn2.running_var"])
            # skip connection + relu
            x = relu(x + residual)

        # ── Policy head ──
        p = conv2d(x, w["policy_conv.weight"], padding=0)
        p = batch_norm(p, w["policy_bn.weight"], w["policy_bn.bias"],
                       w["policy_bn.running_mean"], w["policy_bn.running_var"])
        p = relu(p)
        N = p.shape[0]
        p = p.reshape(N, -1)  # (N, 2*64) = (N, 128)
        p = linear(p, w["policy_fc.weight"], w["policy_fc.bias"])  # (N, 4288)

        # ── Value head ──
        v = conv2d(x, w["value_conv.weight"], padding=0)
        v = batch_norm(v, w["value_bn.weight"], w["value_bn.bias"],
                       w["value_bn.running_mean"], w["value_bn.running_var"])
        v = relu(v)
        v = v.reshape(N, -1)  # (N, 64)
        # No dropout in eval mode
        v = relu(linear(v, w["value_fc1.weight"], w["value_fc1.bias"]))  # (N, 64)
        v = np.tanh(linear(v, w["value_fc2.weight"], w["value_fc2.bias"]))  # (N, 1)

        if N == 1:
            return p[0], float(v[0, 0])
        else:
            return p, v[:, 0]

    # ── I/O ──

    @classmethod
    def from_npz(cls, path: str | Path) -> "NumpyNetwork":
        """Load weights from a .npz file (no torch needed)."""
        data = np.load(str(path))
        weights = {k: data[k] for k in data.files}
        return cls(weights)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "NumpyNetwork":
        """Load weights from a PyTorch .pt checkpoint (requires torch)."""
        weights = load_weights_from_checkpoint(str(path))
        return cls(weights)


# ── Checkpoint conversion ────────────────────────────────────────────────

def load_weights_from_checkpoint(path: str) -> dict[str, np.ndarray]:
    """Extract numpy arrays from a PyTorch checkpoint.

    Requires torch (import only here, not at module level).
    """
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt["network_state"]
    weights = {}
    for key, tensor in state_dict.items():
        # Skip num_batches_tracked — not used in eval-mode BN
        if "num_batches_tracked" in key:
            continue
        weights[key] = tensor.cpu().float().numpy()
    return weights


def convert_checkpoint_to_npz(
    pt_path: str,
    npz_path: str | None = None,
) -> str:
    """Convert a PyTorch checkpoint to .npz format.

    Args:
        pt_path: path to .pt checkpoint
        npz_path: output path; defaults to same directory as pt_path

    Returns:
        Path to the saved .npz file.
    """
    weights = load_weights_from_checkpoint(pt_path)
    if npz_path is None:
        npz_path = str(Path(pt_path).with_suffix(".npz"))
    np.savez_compressed(npz_path, **weights)
    print(f"Saved {len(weights)} weight arrays to {npz_path}")
    # Print file size
    size_mb = Path(npz_path).stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")
    return npz_path


# ── Conversion + verification when run as script ────────────────────────

if __name__ == "__main__":
    import sys
    import os

    # Default paths
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent.parent
    DEFAULT_PT = REPO_ROOT / "checkpoints" / "on_policy_fix_20260308" / "best.pt"
    DEFAULT_NPZ = SCRIPT_DIR / "model_weights.npz"

    pt_path = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_PT)
    npz_path = sys.argv[2] if len(sys.argv) > 2 else str(DEFAULT_NPZ)

    if not Path(pt_path).exists():
        print(f"Checkpoint not found: {pt_path}")
        sys.exit(1)

    # ── Convert ──
    print(f"Converting {pt_path} -> {npz_path}")
    convert_checkpoint_to_npz(pt_path, npz_path)

    # ── Verify: compare numpy vs torch output ──
    print("\nVerifying numpy vs PyTorch outputs...")
    import torch

    # Add repo root to path for imports
    sys.path.insert(0, str(REPO_ROOT))
    from davechess.engine.network import DaveChessNetwork

    # Load PyTorch model
    net_torch, _ = DaveChessNetwork.from_checkpoint(pt_path, device="cpu")
    net_torch.eval()

    # Load numpy model
    net_np = NumpyNetwork.from_npz(npz_path)

    # Random test input
    np.random.seed(42)
    test_input = np.random.randn(1, 18, 8, 8).astype(np.float32)

    # PyTorch inference
    with torch.no_grad():
        t_input = torch.from_numpy(test_input)
        t_policy, t_value = net_torch(t_input)
        torch_policy = t_policy[0].numpy()
        torch_value = t_value.item()

    # Numpy inference
    np_policy, np_value = net_np.predict(test_input[0])

    # Compare
    policy_diff = np.max(np.abs(torch_policy - np_policy))
    value_diff = abs(torch_value - np_value)

    print(f"  Policy max abs diff: {policy_diff:.2e}")
    print(f"  Value abs diff:      {value_diff:.2e}")
    print(f"  PyTorch value:       {torch_value:.6f}")
    print(f"  Numpy value:         {np_value:.6f}")

    # Check top-5 policy indices match
    torch_top5 = np.argsort(torch_policy)[-5:][::-1]
    np_top5 = np.argsort(np_policy)[-5:][::-1]
    print(f"  PyTorch top-5 indices: {torch_top5}")
    print(f"  Numpy   top-5 indices: {np_top5}")

    if policy_diff < 1e-4 and value_diff < 1e-5:
        print("\nVERIFICATION PASSED: outputs match within tolerance.")
    elif policy_diff < 1e-2 and value_diff < 1e-3:
        print("\nVERIFICATION WARNING: outputs are close but not exact "
              "(likely float32 accumulation differences).")
    else:
        print("\nVERIFICATION FAILED: outputs differ significantly!")
        sys.exit(1)
