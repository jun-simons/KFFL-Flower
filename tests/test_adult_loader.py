import numpy as np
import torch

from flwr_datasets.partitioner import IidPartitioner

from kffl.data.adult import load_adult_client_loaders


class DummyContext:
    """Minimal stand-in for Flower Context for data loading tests."""
    def __init__(self, partition_id: int):
        self.node_config = {"partition-id": partition_id}
        self.state = {}


def inspect_one_client(partition_id: int, num_partitions: int = 10):
    context = DummyContext(partition_id)

    partitioner = IidPartitioner(num_partitions=num_partitions)

    loaders = load_adult_client_loaders(
        context=context,
        num_partitions=num_partitions,
        partitioner=partitioner,
        batch_size=64,
        fair_batch_size=256,
        seed=42,
    )

    art = loaders.artifacts
    print(f"\n=== Client {partition_id} ===")
    print(f"x_main_dim={art.x_main_dim}, s_dim={art.s_dim}")

    
    x_main, s, y = next(iter(loaders.trainloader))
    print("train batch shapes:", x_main.shape, s.shape, y.shape)
    assert x_main.ndim == 2
    assert s.ndim == 2
    assert y.ndim == 1
    assert x_main.shape[0] == s.shape[0] == y.shape[0]
    assert x_main.shape[1] == art.x_main_dim
    assert s.shape[1] == art.s_dim

    # Check values are finite
    assert torch.isfinite(x_main).all()
    assert torch.isfinite(s).all()
    assert torch.isfinite(y).all()

    y_unique = torch.unique(y).cpu().numpy()
    print("y unique (sample):", y_unique[:10])
    assert np.all(np.isin(y_unique, [0.0, 1.0])), "y is not binary 0/1"
   
    # s = [race_onehot..., is_male]
    s_row_sums = s.sum(dim=1).cpu().numpy()
    # race_onehot sum is 1; is_male adds 0/1, so totals are 1 or 2
    assert s_row_sums.min() >= 1.0 - 1e-5
    assert s_row_sums.max() <= 2.0 + 1e-5
    # Check split sizes
    n_train = len(loaders.trainloader.dataset)
    n_test = len(loaders.testloader.dataset)
    print("sizes:", n_train, n_test, "test_frac=", n_test / (n_train + n_test))
    assert n_train > 0 and n_test > 0

    return loaders


def test_split_is_stable(partition_id: int, num_partitions: int = 10):
    """Same partition-id should yield identical split across repeated calls (same seed)."""
    c1 = DummyContext(partition_id)
    c2 = DummyContext(partition_id)

    partitioner = IidPartitioner(num_partitions=num_partitions)

    a = load_adult_client_loaders(context=c1, num_partitions=num_partitions, partitioner=partitioner, seed=42)
    b = load_adult_client_loaders(context=c2, num_partitions=num_partitions, partitioner=partitioner, seed=42)

    # Compare a small fingerprint of the first batch to ensure identical data ordering not guaranteed (shuffle=True),
    # but fairnessloader is shuffle=False so compare that

    xma, sa, ya = next(iter(a.fairnessloader))
    xmb, sb, yb = next(iter(b.fairnessloader))

    fa = torch.sum(xma[:10]).item() + torch.sum(sa[:10]).item() + torch.sum(ya[:10]).item()
    fb = torch.sum(xmb[:10]).item() + torch.sum(sb[:10]).item() + torch.sum(yb[:10]).item()
    print("fairness fingerprint:", fa, fb)
    assert abs(fa - fb) < 1e-6, "Splits/loaders are not stable across runs for the same partition-id"


if __name__ == "__main__":
    # Smoke test two different clients
    inspect_one_client(0)
    inspect_one_client(1)

    # Stability test for a single client
    test_split_is_stable(0)

    print("\n✅ smoke_test_adult passed")
