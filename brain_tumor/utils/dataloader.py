from torch.utils.data import DataLoader


def loading_data(exp_specs, batch_size: int, num_workers: int = 0) -> DataLoader:
    if exp_specs["task"] == "classification":
        from brain_tumor.datasets import SNNClassificationDataset as BraTDataset
    elif exp_specs["task"] == "segmentation":
        from brain_tumor.datasets import BraTSegmentationDataset as BraTDataset
    else:
        raise NotImplementedError
    dataset = BraTDataset

    train_dataset = dataset(
        exp_specs=exp_specs,
        root=exp_specs["root"],
        img_dim=exp_specs["img_dim"],
        mri_type=exp_specs["mri_type"],
        train=True,
    )
    val_dataset = dataset(
        exp_specs=exp_specs,
        root=exp_specs["root"],
        img_dim=exp_specs["img_dim"],
        mri_type=exp_specs["mri_type"],
        train=False,
    )

    test_dataset = dataset(
        exp_specs=exp_specs,
        root=exp_specs["root"],
        img_dim=exp_specs["img_dim"],
        mri_type=exp_specs["mri_type"],
        train=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=train_dataset._collate_fn,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=val_dataset._collate_fn,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=test_dataset._collate_fn,
    )

    return train_dataloader, val_dataloader, test_dataloader
