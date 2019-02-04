from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os


def load_data_train(img_size):
	transformers = transforms.Compose([
					transforms.Resize((img_size, img_size)),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
	return transformers


def load_data_test(img_size):
	transformers = transforms.Compose([
					transforms.Resize((img_size, img_size)),
					transforms.ToTensor(),
					transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
	return transformers


def run_loader(ftype, path, batch_size, imageSize, shuffle):
	if ftype is 'test':
		transformers = load_data_test(imageSize)
	else:
		transformers = load_data_train(imageSize)
	set_data = datasets.ImageFolder(root=path, transform=transformers)
	return DataLoader(set_data, batch_size=batch_size, shuffle=shuffle, num_workers=2)


def calculate_data_size(path):
	size = 0
	for i in os.listdir(path):
		size += len(os.listdir('{}/{}'.format(path, i)))
	return size
