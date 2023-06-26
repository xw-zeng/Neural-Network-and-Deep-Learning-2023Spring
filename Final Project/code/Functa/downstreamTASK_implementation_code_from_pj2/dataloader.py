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
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2)
    
    sample_index = [i for i in range(9000)] #假设取前9000个训练数据
    X_train = []
    y_train = []
    for i in sample_index:
        X = train_set[i][0]
        X_train.append(X)
        y = train_set[i][1]
        y_train.append(y)

    sampled_train_data = [(X, y) for X, y in zip(X_train, y_train)] #包装为数据对
    train_loader = torch.utils.data.DataLoader(sampled_train_data, batch_size=16, shuffle=True)

    
    # train_set = torchvision.datasets.CIFAR10(root=cache_path, train=True, download=True, transform=train_transform)
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2)
    return (sampled_train_data, train_loader)

def load_testing_data(batch_size = 16, cache_path="./data"):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_set = torchvision.datasets.MNIST(root=cache_path, train=False, download=True, transform=test_transform)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=True, num_workers=2)
    
    sample_index = [i for i in range(1000)] #假设取前1000个训练数据
    X_test = []
    y_test = []
    for i in sample_index:
        X = test_set[i][0]
        X_test.append(X)
        y = test_set[i][1]
        y_test.append(y)

    sampled_test_data = [(X, y) for X, y in zip(X_test, y_test)] #包装为数据对
    test_loader = torch.utils.data.DataLoader(sampled_test_data, batch_size=16, shuffle=True)
    
    # test_set = torchvision.datasets.CIFAR10(root=cache_path, train=False, download=True, transform=test_transform)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=True, num_workers=2)
    return test_loader
    