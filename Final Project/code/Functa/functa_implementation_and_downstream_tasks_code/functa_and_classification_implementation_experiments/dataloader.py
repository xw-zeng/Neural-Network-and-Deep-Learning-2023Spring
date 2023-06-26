import torchvision
import torchvision.transforms as transforms
import torch

def load_training_set(batch_size = 16, cache_path="./data"):
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.RandomCrop(32, padding=5),
        # transforms.RandomRotation(45),
        # transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
        # transforms.RandomAutocontrast(p=0.1),
        # transforms.RandomGrayscale(p=0.1),
        # transforms.RandomInvert(p=0.1),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_set = torchvision.datasets.MNIST(root=cache_path, train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2)
    
    # train_set = torchvision.datasets.CIFAR10(root=cache_path, train=True, download=True, transform=train_transform)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2)
    return (train_set, train_loader)

def load_testing_data(batch_size = 16, cache_path="./data"):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_set = torchvision.datasets.MNIST(root=cache_path, train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=True, num_workers=2)
    
    # test_set = torchvision.datasets.CIFAR10(root=cache_path, train=False, download=True, transform=test_transform)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=True, num_workers=2)
    return test_loader
    