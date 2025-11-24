# /// script
# requires-python = "==3.13"
# dependencies = [
#    "flyte>=2.0.0b26",
#    "torch==2.7.1",
#    "torchvision==0.22.1",
#    "lightning==2.5.2",
#    "unionai-reuse==0.1.3",
#    "wandb==0.21.0",
# ]
# ///

import asyncio
import os

import lightning as L
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import flyte
import flyte.io

image = flyte.Image.from_uv_script(__file__, name="optimizer-gpu")

gpu_env = flyte.TaskEnvironment(
    name="gpu_env",
    resources=flyte.Resources(cpu=6, memory="4Gi", gpu="L4:1"),
    image=image,
    secrets=flyte.Secret(key="NIELS_WANDB_API_KEY", as_env_var="WANDB_API_KEY"),
)

driver = flyte.TaskEnvironment(
    name="gridsearch_driver",
    resources=flyte.Resources(cpu=4, memory="1Gi"),
    image=image,
    depends_on=[gpu_env],
)


class MNISTAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        x, _y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        wandb.run.log({"train_loss": loss})
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, root_dir, batch_size=64, dataloader_num_workers=0):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.dataloader_num_workers = dataloader_num_workers

    def prepare_data(self):
        MNIST(self.root_dir, train=True, download=True)

    def setup(self, stage=None):
        self.train_dataset = MNIST(
            self.root_dir,
            train=True,
            download=False,
            transform=ToTensor(),
        )

    def train_dataloader(self):
        persistent_workers = self.dataloader_num_workers > 0
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.dataloader_num_workers,
            persistent_workers=persistent_workers,
            pin_memory=True,
            shuffle=True,
        )


@gpu_env.task
async def train_model(
    sweep_name: str,
    batch_size: int,
    dataloader_num_workers: int = 1,
) -> tuple[flyte.io.Dir, float]:
    """Train an autoencoder model on the MNIST."""
    run = wandb.init(project="flyte-ml-optimizer", name=sweep_name, id=sweep_name)

    encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
    decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))
    autoencoder = MNISTAutoEncoder(encoder, decoder)

    root_dir = os.getcwd()
    data = MNISTDataModule(
        root_dir,
        batch_size=batch_size,
        dataloader_num_workers=dataloader_num_workers,
    )

    model_dir = os.path.join(root_dir, "model")
    trainer = L.Trainer(
        default_root_dir=model_dir,
        max_epochs=1,
        accelerator="gpu",
        precision="16-mixed",
    )

    if batch_size == 16:
        raise RuntimeError("Oops! Something went wrong...")

    trainer.fit(model=autoencoder, datamodule=data)

    train_loss = trainer.callback_metrics["train_loss"]
    run.log({"final_train_loss": train_loss, "batch_size": batch_size})
    run.finish()

    dir = await flyte.io.Dir.from_local(local_path=str(model_dir))
    return dir, float(train_loss.item())


@driver.task
async def gridsearch(
    sweep_name: str,
    batch_sizes: list[int],
) -> tuple[flyte.io.Dir, float]:
    results = []
    for i, batch_size in enumerate(batch_sizes):
        results.append(train_model.override(short_name=f"train-model-bs-{batch_size}")(f"{sweep_name}-{i}", batch_size))

    results = await asyncio.gather(*results)
    best_model, best_train_loss = min(results, key=lambda x: x[1])
    return best_model, best_train_loss


if __name__ == "__main__":
    from datetime import datetime

    flyte.init_from_config()
    sweep_name = f"hpo-gpu-sweep-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    run = flyte.run(gridsearch, sweep_name, batch_sizes=[4, 8, 16, 32])
    print(run.url)
