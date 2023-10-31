import torch
import torch.utils.data


def get_dataloader(full_dataset, batch_size=2**8, split="train", subset_fraction=1):
    """[TODO:description]

    Args:
        full_dataset (dataset): full data set
        batch_size ([TODO:parameter]): size of batches in loader
        split (str): which split to load (train, val, or test)
        subset_fraction (int): Load only every nth sample

    Returns:
        Torch dataloader with a Torch subset (even if subset_fraction=1) of the full dataset
        as the dataloader.dataset attribute
    """
    "Get a dataloader for training or testing"

    assert isinstance(full_dataset, torch.utils.data.Dataset)
    is_train = split == "train"

    sub_dataset = torch.utils.data.Subset(
        full_dataset, indices=range(0, len(full_dataset), subset_fraction)
    )

    loader = torch.utils.data.DataLoader(
        dataset=sub_dataset,
        batch_size=batch_size,
        shuffle=is_train,
        pin_memory=True,
        num_workers=0,
    )
    return loader
