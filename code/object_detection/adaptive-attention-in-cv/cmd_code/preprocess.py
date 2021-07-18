import os
import torch
import pickle
from torchvision import datasets, transforms
from typing import Any, Callable, Optional, Tuple
# NOTE: Mean and std used for normalization are known stats from the distribution of each dataset
import torch.utils.data as data
from typing import Any, Callable, List, Optional, Tuple


class VisionDataset(data.Dataset):
    _repr_indent = 4

    def __init__(
            self,
            root: str,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can "
                             "be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self) -> str:
        return ""


class StandardTransform(object):
    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input: Any, target: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transform: ")

        return '\n'.join(body)

class Dataset(VisionDataset):
  'Characterizes a dataset for PyTorch'
  def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(Dataset, self).__init__(root, transform=transform,
                                      target_transform=target_transform)
        'Initialization'
        self.data = []
        self.targets = []
        all_data = pickle.load(open('/content/drive/MyDrive/fashion_data.pickle', 'rb'))
        for entity in all_data:
          self.data.append(entity[0])
          self.targets.append(entity[1])

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.targets)

  def __getitem__(self, index):
        'Generates one sample of data'
        img, target = self.data[index], self.targets[index]
        # Select sample
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def load_data(args):
    print('Load Dataset :: {}'.format(args.dataset))
    if args.dataset == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            )
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            )
        ])

        train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
        train_len = int(len(train_data)*0.9)
        val_len = len(train_data) - train_len
        print('Len Train: {}, Len Valid: {}'.format(train_len,val_len))
        train_set, valid_set = torch.utils.data.random_split(train_data, [train_len, val_len])
        valid_set.transform = transform_test #Don't want to apply flips and random crops to this

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=False, transform=transform_test),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

    elif args.dataset == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4865, 0.4409),
                std=(0.2673, 0.2564, 0.2762)
            ),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4865, 0.4409),
                std=(0.2673, 0.2564, 0.2762)
            ),
        ])

        #train_data = datasets.CIFAR100('data', train=True, download=True, transform=transform_train)
        train_data = Dataset('data', train=True, transform=transform_train)
        actual_to_be_used = int(len(train_data) * args.subset)
        train_data, _ = torch.utils.data.random_split(train_data, [actual_to_be_used, len(train_data) - actual_to_be_used])

        train_len = int(len(train_data)*0.9)
        val_len = len(train_data) - train_len
        print('Len Train: {}, Len Valid: {}'.format(train_len,val_len))
        train_set, valid_set = torch.utils.data.random_split(train_data, [train_len, val_len])
        valid_set.transform = transform_test #Don't want to apply flips and random crops to this

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        test_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

    elif args.dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.1307,),
                std=(0.3081,)
            )
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True, transform=transform),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

    elif args.dataset == 'TinyImageNet':
        # We use the normalization stats of the full ImageNet dataset as an estimate for the stats of the
        # TinyImageNet dataset
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])

        # Only the training data has labels so we will split it up to make training, testing, and validation sets
        train_data = datasets.ImageFolder('./datasets/processed-tiny-imagenet', transform=transform_train)

        train_len = int(len(train_data)*0.8)
        val_len = int(len(train_data)*0.1)
        test_len = int(len(train_data)*0.1)
        train_set, valid_set, test_set = torch.utils.data.random_split(train_data, [train_len, val_len, test_len])
        
        # test_len = int(len(train_set)*0.1)
        # new_train_len = len(train_set) - test_len
        # train_set, test_set = torch.utils.data.random_split(train_set, [new_train_len, test_len])
        
        #Don't want to apply flips and random crops to this
        valid_set.transform = transform_test
        test_set.transform = transform_test

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        print('TinyImageNet Loader')
        print(train_loader)

    return train_loader, valid_loader, test_loader


#This is just for testing purposes
class Args:
    def __init__(self):
        self.batch_size = 32
        self.num_workers = 1
        self.dataset = 'CIFAR10'

if __name__ == '__main__':

    #need to split the training set into train/valid
    args = Args()
    train, valid, test = load_data(args)
