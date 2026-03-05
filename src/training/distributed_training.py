"""
PyTorch Distributed Data Parallel (DDP) Training Example

Demonstrates multi-GPU and multi-node training with PyTorch's native
DistributedDataParallel module.  Covers:
  - Process group initialization (NCCL backend)
  - Dataset sharding with DistributedSampler
  - DDP model wrapping with gradient synchronization
  - Mixed-precision training with torch.cuda.amp
  - Checkpointing and graceful shutdown
  - FSDP (Fully Sharded Data Parallel) comparison notes

Usage (single node, 4 GPUs):
    torchrun --nproc_per_node=4 src/training/distributed_training.py

Usage (multi-node, 2 nodes x 4 GPUs):
    # On node 0:
    torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
             --master_addr=10.0.0.1 --master_port=29500 \
             src/training/distributed_training.py

    # On node 1:
    torchrun --nnodes=2 --node_rank=1 --nproc_per_node=4 \
             --master_addr=10.0.0.1 --master_port=29500 \
             src/training/distributed_training.py

Requirements:
    pip install torch torchvision
"""

import os
import sys
import time
import logging
import argparse
from datetime import timedelta
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.cuda.amp import GradScaler, autocast

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Rank %(process)d] %(levelname)s - %(message)s",
)
logger = logging.getLogger("distributed_training")


# ---------------------------------------------------------------------------
# Synthetic Dataset (replace with your real dataset)
# ---------------------------------------------------------------------------
class SyntheticImageDataset(Dataset):
    """
    Generates random image-like tensors for benchmarking.
    Replace with torchvision.datasets.ImageFolder or a custom dataset.
    """

    def __init__(self, num_samples: int = 10000, image_size: int = 224, num_classes: int = 100):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = torch.randn(3, self.image_size, self.image_size)
        label = idx % self.num_classes
        return image, label


# ---------------------------------------------------------------------------
# Simple CNN Model (replace with your architecture)
# ---------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """
    A straightforward CNN for demonstration purposes.
    In production, use ResNet, EfficientNet, ViT, etc.
    """

    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Distributed Training Helper
# ---------------------------------------------------------------------------
class DistributedTrainer:
    """
    Manages the full lifecycle of a DDP training run:
      1. Process group setup
      2. Model wrapping
      3. Training loop with mixed precision
      4. Checkpointing
      5. Cleanup
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.rank: int = 0
        self.local_rank: int = 0
        self.world_size: int = 1
        self.device: torch.device = torch.device("cpu")
        self.model: Optional[DDP] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scaler: Optional[GradScaler] = None
        self.criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # 1. Process group initialization
    # ------------------------------------------------------------------
    def setup(self):
        """Initialize the distributed process group (NCCL backend)."""
        # torchrun sets these environment variables automatically
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        if self.world_size > 1:
            dist.init_process_group(
                backend="nccl",            # Use NCCL for GPU communication
                timeout=timedelta(minutes=10),
            )
            logger.info(
                "Process group initialized: rank=%d, local_rank=%d, world_size=%d",
                self.rank, self.local_rank, self.world_size,
            )
        else:
            logger.info("Running in single-GPU mode (world_size=1).")

        # Pin this process to its local GPU
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            logger.warning("CUDA not available. Training on CPU (very slow).")
            self.device = torch.device("cpu")

    # ------------------------------------------------------------------
    # 2. Model, optimizer, scaler
    # ------------------------------------------------------------------
    def build_model(self):
        """Create the model, wrap it in DDP, and set up the optimizer."""
        model = SimpleCNN(num_classes=self.args.num_classes).to(self.device)

        if self.world_size > 1:
            self.model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,  # Set True only if needed
            )
        else:
            self.model = model

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr * self.world_size,  # Linear scaling rule
            weight_decay=self.args.weight_decay,
        )

        # Mixed-precision scaler (FP16/BF16)
        if self.args.mixed_precision and torch.cuda.is_available():
            self.scaler = GradScaler()
            logger.info("Mixed-precision training enabled (FP16).")

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Model parameters: %s", f"{total_params:,}")

    # ------------------------------------------------------------------
    # 3. Data loading with DistributedSampler
    # ------------------------------------------------------------------
    def get_dataloader(self) -> DataLoader:
        """Create a DataLoader with DistributedSampler for sharding."""
        dataset = SyntheticImageDataset(
            num_samples=self.args.num_samples,
            num_classes=self.args.num_classes,
        )

        sampler = None
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )

        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        logger.info(
            "DataLoader created: %d samples, batch_size=%d, %d batches/epoch",
            len(dataset), self.args.batch_size, len(loader),
        )
        return loader

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    def train_epoch(self, loader: DataLoader, epoch: int) -> dict:
        """Run one training epoch and return metrics."""
        self.model.train()

        # Update sampler epoch for proper shuffling across ranks
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                # Mixed-precision forward pass
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Log every N batches (only from rank 0)
            if self.rank == 0 and (batch_idx + 1) % self.args.log_interval == 0:
                elapsed = time.time() - start_time
                samples_per_sec = total / elapsed * self.world_size
                logger.info(
                    "Epoch %d [%d/%d] loss=%.4f acc=%.2f%% throughput=%.0f samples/s",
                    epoch,
                    batch_idx + 1,
                    len(loader),
                    total_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    samples_per_sec,
                )

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(loader)
        accuracy = 100.0 * correct / total

        return {
            "epoch": epoch,
            "loss": avg_loss,
            "accuracy": accuracy,
            "epoch_time": epoch_time,
            "throughput": total / epoch_time * self.world_size,
        }

    # ------------------------------------------------------------------
    # 5. Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self, epoch: int, metrics: dict):
        """Save a training checkpoint (only on rank 0)."""
        if self.rank != 0:
            return

        checkpoint_dir = self.args.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_state = (
            self.model.module.state_dict()
            if isinstance(self.model, DDP)
            else self.model.state_dict()
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "world_size": self.world_size,
        }

        path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pt")
        torch.save(checkpoint, path)
        logger.info("Checkpoint saved: %s", path)

    # ------------------------------------------------------------------
    # 6. Full training run
    # ------------------------------------------------------------------
    def train(self):
        """Execute the complete training run."""
        loader = self.get_dataloader()
        best_loss = float("inf")

        logger.info(
            "Starting training: %d epochs, %d GPU(s)",
            self.args.epochs, self.world_size,
        )

        for epoch in range(1, self.args.epochs + 1):
            metrics = self.train_epoch(loader, epoch)

            if self.rank == 0:
                logger.info(
                    "Epoch %d complete: loss=%.4f acc=%.2f%% time=%.1fs throughput=%.0f samples/s",
                    epoch,
                    metrics["loss"],
                    metrics["accuracy"],
                    metrics["epoch_time"],
                    metrics["throughput"],
                )

                # Save checkpoint if loss improved
                if metrics["loss"] < best_loss:
                    best_loss = metrics["loss"]
                    self.save_checkpoint(epoch, metrics)

            # Synchronize before next epoch
            if self.world_size > 1:
                dist.barrier()

        if self.rank == 0:
            logger.info("Training complete. Best loss: %.4f", best_loss)

    # ------------------------------------------------------------------
    # 7. Cleanup
    # ------------------------------------------------------------------
    def cleanup(self):
        """Destroy the process group."""
        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Process group destroyed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyTorch DDP Distributed Training Example",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num-samples", type=int, default=10000, help="Synthetic dataset size")
    parser.add_argument("--num-classes", type=int, default=100, help="Number of classes")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable FP16 mixed precision")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N batches")
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = DistributedTrainer(args)

    try:
        trainer.setup()
        trainer.build_model()
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        sys.exit(1)
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
