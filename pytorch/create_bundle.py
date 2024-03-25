import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, ConfusionMatrix, RunningAverage
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers import ProgressBar


# CNN model class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(64 * 6 * 6, 600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(600, 120)
        self.fc3 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = x.view(-1, 64 * 6 * 6)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":

    # Transform to normalize the data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Download and load the training data
    trainset = datasets.FashionMNIST(
        "./data", download=True, train=True, transform=transform
    )
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)

    # Download and load the test data
    validationset = datasets.FashionMNIST(
        "./data", download=True, train=False, transform=transform
    )
    val_loader = DataLoader(validationset, batch_size=64, shuffle=True)

    # Creating model, optimizer, and loss
    model = CNN()

    # Moving model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    # Defining the number of epochs
    epochs = 3

    # Creating trainer and evaluator
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    metrics = {
        "accuracy": Accuracy(),
        "nll": Loss(criterion),
        "cm": ConfusionMatrix(num_classes=10),
    }
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

    # Early stopping based on validation loss
    def score_function(engine):
        val_loss = engine.state.metrics["nll"]
        return -val_loss

    handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
    val_evaluator.add_event_handler(Events.COMPLETED, handler)

    pbar = ProgressBar(persist=True, bar_format="")
    pbar.attach(trainer, ["loss"])
    trainer.run(train_loader, max_epochs=epochs)

    torch.save(
        model.state_dict(),
        "data/model.pkl",
    )
