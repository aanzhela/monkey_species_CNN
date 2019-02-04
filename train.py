import torch
from model import ConvNet
from load import run_loader, calculate_data_size
from tqdm import tqdm


class Training:
    def __init__(self, epoch, learningRate, batchSize, imageSize, L2Rate, trainPath):
        super(Training, self).__init__()
        self.epoch = epoch
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.imageSize = imageSize
        self.L2Rate = L2Rate
        self.trainPath = trainPath
        self.data_size = calculate_data_size(self.trainPath)
        self.num_batches = self.data_size // batchSize
        self.data_loader = run_loader('train', trainPath, batchSize, imageSize, shuffle=True)
        self.model = ConvNet(10)
        self.train()

    def train(self):
        self.model.train()

        crossentropy = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.L2Rate)

        for epoch in range(self.epoch):
            epoch_loss = 0
            epoch_acc = 0
            for X, y in tqdm(self.data_loader):
                optimizer.zero_grad()
                out = self.model(X)

                loss = crossentropy(out, y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()  # makes it to python float
                predictions = torch.argmax(out, 1)
                epoch_acc += torch.sum(predictions == y).item()

            epoch_loss = epoch_loss / self.num_batches
            epoch_acc = epoch_acc / self.data_size
            print(f"Epoch {epoch}:", "ACC:", epoch_acc, "LOSS:", epoch_loss)

            torch.save(self.model.state_dict(), f"Trained/Model_{epoch}.model")


if __name__ == "__main__":
    Training(200, 0.001, 64, 64, 0, "Data/monkey_species_train")
