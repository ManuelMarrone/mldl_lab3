import torch
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

def get_dataloaders():
    # ======== DATASET E DATALOADER =========
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    tiny_imagenet_dataset_train = ImageFolder(
        root='tiny-imagenet/tiny-imagenet-200/train',
        transform=transform
    )
    tiny_imagenet_dataset_val = ImageFolder(
        root='tiny-imagenet/tiny-imagenet-200/val',
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        tiny_imagenet_dataset_train, batch_size=32, shuffle=True, num_workers=8
    )
    val_loader = torch.utils.data.DataLoader(
        tiny_imagenet_dataset_val, batch_size=32, shuffle=False
    )

    print("Dataset pronto! Classi:", len(tiny_imagenet_dataset_train.classes))


    return train_loader, val_loader