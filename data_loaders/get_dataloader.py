import torch
import torch.utils.data


def get_dataloader(full_dataset, batch_size=2**8, split="train", subset_fraction=1):
    "Get a dataloader for training or testing"

    assert isinstance(full_dataset, torch.utils.data.Dataset)
    is_train = split == "train"

    if subset_fraction > 1:
        sub_dataset = torch.utils.data.Subset(
            full_dataset, indices=range(0, len(full_dataset), subset_fraction)
        )
    else:
        sub_dataset = full_dataset

    loader = torch.utils.data.DataLoader(
        dataset=sub_dataset,
        batch_size=batch_size,
        shuffle=is_train,
        pin_memory=True,
        num_workers=0,
    )
    return loader
