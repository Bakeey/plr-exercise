from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb
import optuna
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from plr_exercise.models.cnn import Net


def train(args, model, device, train_loader, optimizer, epoch):
    """
    Trains the model for a single epoch.

    Args:
        args: Command-line arguments.
        model (torch.nn.Module): The model to train.
        device (torch.device): The device to train the model on.
        train_loader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        epoch (int): Current epoch number.

    Prints training loss and logs it to wandb at intervals specified by args.log_interval.
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            wandb.log({"training_loss": loss.item()})
            if args.dry_run:
                break


def test(model, device, test_loader, epoch):
    """
    Evaluates the model on the test dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        device (torch.device): The device to evaluate the model on.
        test_loader (DataLoader): DataLoader for the test data.
        epoch (int): Current epoch number.

    Returns:
        float: The average loss of the model on the test dataset.

    Logs the test loss to wandb and prints the test accuracy.
    """
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    wandb.log({"test_loss": test_loss})

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )
    return test_loss


def main():
    """
    Main function to run the training and optimization process.

    Initializes wandb, sets up command-line arguments for training parameters,
    and uses Optuna to find the best learning rate and number of epochs.
    Logs the model's source code as an artifact in wandb.
    """
    # wandb initialization
    training_loss = 0.0
    test_loss = 0.0

    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="plr-exercise",
        # Track hyperparameters and run metadata
        config={
            "training_loss": training_loss,
            "test_loss": test_loss,
        },
        settings=wandb.Settings(code_dir="."),
    )

    """
    # log the code as artifact
    code_artifact = wandb.Artifact('source-code', type='code')
    # Add a directory or specific files to the artifact
    artifact = wandb.Artifact('source-code', type='code')
    artifact.add_file('./scripts/train.py')
    run.log_artifact(artifact)
    """

    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=2, metavar="N", help="number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    args = parser.parse_args()

    # Optuna setup
    study = optuna.create_study(direction="minimize")

    def objective(trial):
        """
        Objective function for Optuna study to optimize hyperparameters.

        Args:
            trial (optuna.trial.Trial): A trial from the study.

        Returns:
            float: The test loss of the model trained with the suggested hyperparameters.

        Suggests learning rates and epochs, trains the model on the MNIST dataset,
        and evaluates it to return the test loss.
        """
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
        epochs = trial.suggest_int("epochs", 1, 20)
        use_cuda = not args.no_cuda and torch.cuda.is_available()

        torch.manual_seed(args.seed)

        if use_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        train_kwargs = {"batch_size": args.batch_size}
        test_kwargs = {"batch_size": args.test_batch_size}
        if use_cuda:
            cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
        dataset2 = datasets.MNIST("../data", train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

        model = Net().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test_loss = test(model, device, test_loader, epoch)

        """
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(args.epochs):
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader, epoch)
            scheduler.step()
        """

        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn.pt")
        return test_loss

    study.optimize(objective, n_trials=100)
    print("Best hyperparameters: ", study.best_params)

    run.finish()


if __name__ == "__main__":
    main()
