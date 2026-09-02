import torch

from heston_fno.data.normalization import StandardScaler


def test_scaler_preserves_unbatched_sample_shape() -> None:
    scaler = StandardScaler(
        mean=torch.zeros(1, 7, 1, 1),
        std=torch.ones(1, 7, 1, 1),
    )
    sample = torch.randn(7, 64, 32)

    transformed = scaler.transform(sample)

    assert transformed.shape == sample.shape
    assert torch.equal(transformed, sample)


def test_scaler_preserves_batched_tensor_shape() -> None:
    scaler = StandardScaler(
        mean=torch.zeros(1, 7, 1, 1),
        std=torch.ones(1, 7, 1, 1),
    )
    batch = torch.randn(16, 7, 64, 32)

    assert scaler.transform(batch).shape == batch.shape
