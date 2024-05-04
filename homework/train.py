from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse


def train(args):
    model = model_factory[args.model]()
    val_loader = load_data("./data/valid")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.1e-4)
    criterion = ClassificationLoss()
    writer = SummaryWriter(log_dir=f'runs/{args.model}_train_10')

    for epoch in range(args.epochs):
        train_loader = load_data("./data/train")
        model.train()
        total_loss = 0
        count = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            count += 1
            if count % 100 == 0:
                writer.add_scalar('Training/Loss', total_loss / count, (epoch + 1) * count)
                
        # Model validation        
        model.eval()
        val_loss, total_val_accuracy, val_total_loss, val_count = 0, 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                val_total_loss += val_loss.item()
                total_val_accuracy += accuracy(outputs, labels)
                val_count += 1
        writer.add_scalar('Validation/Loss', val_total_loss / val_count, (epoch + 1) * val_count)
        writer.add_scalar('Validation/Accuracy', total_val_accuracy / val_count, (epoch + 1) * val_count)
    writer.close()
    save_model(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    parser.add_argument('-e', '--epochs', type=int, default=20)
    args = parser.parse_args()
    train(args)
