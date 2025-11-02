from torchvision import transforms


IMG_SIZE = 224


train_transform = transforms.Compose([
transforms.RandomResizedCrop(IMG_SIZE),
transforms.RandomHorizontalFlip(p=0.5),
transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])




# deterministic transform for inference
test_transform = transforms.Compose([
transforms.Resize((IMG_SIZE, IMG_SIZE)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
