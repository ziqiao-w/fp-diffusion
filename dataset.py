from datasets import load_from_disk
from torchvision.transforms import Compose
# load dataset from the hub

image_size = 32
channels = 3
batch_size = 16
dataset = load_from_disk("./")
from torchvision import transforms
from torch.utils.data import DataLoader

# define image transformations (e.g. using torchvision)
transform = Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)
])


# define function
def transforms(examples):
    examples["pixel_values"] = [transform(image) for image in examples["img"]]
    del examples["img"]
    return examples

transformed_dataset = dataset.with_transform(transforms).remove_columns("label")

# create dataloader
dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)
print("data prepared")