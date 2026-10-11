"""Training callbacks shared by all models."""

import lightning.pytorch as pl
import torch


class MPSMemoryGuard(pl.Callback):
    """Abort when MPS driver memory exceeds the recommended maximum.

    On MPS an oversized batch does not raise OOM: it swaps and slows down by
    more than 100x (docs/feasibility.md). This turns that into a clear error.
    No-op on other devices.
    """

    def __init__(self, fraction: float = 1.0) -> None:
        self.fraction = fraction

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if pl_module.device.type != "mps":
            return
        used = torch.mps.driver_allocated_memory()
        limit = torch.mps.recommended_max_memory() * self.fraction
        if used > limit:
            raise MemoryError(
                f"MPS memory {used / 2**30:.1f} GiB exceeds limit {limit / 2**30:.1f} GiB "
                "(the system would start swapping). Lower the batch size and raise "
                "trainer.accumulate_grad_batches, or run on a rented GPU."
            )
