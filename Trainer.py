import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import time
from datetime import datetime
import os

def get_device():
    """Get the best available device for training"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')  # Apple Silicon GPU
    else:
        return torch.device('cpu')

class Trainer:
    def __init__(self, model, train_loader, test_loader, configs):
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.configs = configs
        self.optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=configs.step_size, gamma=configs.gamma)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.writer = SummaryWriter(
            'tensorboard/' + configs.llm_type + configs.dataset + datetime.now().strftime("%Y%m%d%H%M%S"))
        self.best_test_accuracy = 0
        self.best_epoch = 0

    def train(self):
        self.model.train()
        total_loss = 0
        total_accuracy = MulticlassAccuracy(num_classes=self.configs.num_class).to(self.device)
        total_precision = MulticlassPrecision(num_classes=self.configs.num_class).to(self.device)
        total_recall = MulticlassRecall(num_classes=self.configs.num_class).to(self.device)
        total_f1 = MulticlassF1Score(num_classes=self.configs.num_class).to(self.device)

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.float().to(self.device), target.long().to(self.device)

            # Use appropriate autocast for the device
            if self.device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    output_y = self.model(data)
                    _, output = torch.max(output_y, 1)
                    loss = self.loss_function(output_y, target)
            else:
                # For MPS and CPU, don't use autocast or use different settings
                output_y = self.model(data)
                _, output = torch.max(output_y, 1)
                loss = self.loss_function(output_y, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_accuracy(output, target)
            total_precision(output, target)
            total_recall(output, target)
            total_f1(output, target)

        avg_loss = total_loss / len(self.train_loader)
        avg_accuracy = total_accuracy.compute()
        avg_precision = total_precision.compute()
        avg_recall = total_recall.compute()
        avg_f1 = total_f1.compute()

        # Log training metrics
        self.writer.add_scalar('Train/Loss', avg_loss, global_step=self.scheduler.last_epoch)
        self.writer.add_scalar('Train/Accuracy', avg_accuracy, global_step=self.scheduler.last_epoch)
        self.writer.add_scalar('Train/Precision', avg_precision, global_step=self.scheduler.last_epoch)
        self.writer.add_scalar('Train/Recall', avg_recall, global_step=self.scheduler.last_epoch)
        self.writer.add_scalar('Train/F1 Score', avg_f1, global_step=self.scheduler.last_epoch)

        print(f'Epoch {self.scheduler.last_epoch} - Train Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1 Score: {avg_f1:.4f}')

    def test(self):
        self.model.eval()
        total_loss = 0
        total_accuracy = MulticlassAccuracy(num_classes=self.configs.num_class).to(self.device)
        total_precision = MulticlassPrecision(num_classes=self.configs.num_class).to(self.device)
        total_recall = MulticlassRecall(num_classes=self.configs.num_class).to(self.device)
        total_f1 = MulticlassF1Score(num_classes=self.configs.num_class).to(self.device)

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.float().to(self.device), target.long().to(self.device)

                # Use appropriate autocast for the device
                if self.device.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        output_y = self.model(data)
                        _, output = torch.max(output_y, 1)
                        loss = self.loss_function(output_y, target)
                else:
                    # For MPS and CPU, don't use autocast
                    output_y = self.model(data)
                    _, output = torch.max(output_y, 1)
                    loss = self.loss_function(output_y, target)

                total_loss += loss.item()
                total_accuracy(output, target)
                total_precision(output, target)
                total_recall(output, target)
                total_f1(output, target)

        avg_loss = total_loss / len(self.test_loader)
        avg_accuracy = total_accuracy.compute()
        avg_precision = total_precision.compute()
        avg_recall = total_recall.compute()
        avg_f1 = total_f1.compute()

        # Log test metrics
        self.writer.add_scalar('Test/Loss', avg_loss, global_step=self.scheduler.last_epoch)
        self.writer.add_scalar('Test/Accuracy', avg_accuracy, global_step=self.scheduler.last_epoch)
        self.writer.add_scalar('Test/Precision', avg_precision, global_step=self.scheduler.last_epoch)
        self.writer.add_scalar('Test/Recall', avg_recall, global_step=self.scheduler.last_epoch)
        self.writer.add_scalar('Test/F1 Score', avg_f1, global_step=self.scheduler.last_epoch)

        print(f'Epoch {self.scheduler.last_epoch} - Test Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1 Score: {avg_f1:.4f}')

        # Checkpoint the best model
        if avg_accuracy > self.best_test_accuracy:
            self.best_test_accuracy = avg_accuracy
            self.best_epoch = self.scheduler.last_epoch
            checkpoint_path = f"{self.configs.path}/best_model_epoch_{self.best_epoch}.pt"
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"New best model saved at {checkpoint_path} with accuracy: {avg_accuracy:.4f}")

    def run(self):
        is_exist = os.path.exists(self.configs.path)
        if not is_exist:
            os.makedirs(self.configs.path)
        for epoch in range(self.configs.epochs):
            start_time = time.time()
            print(f'Starting Epoch {epoch + 1}/{self.configs.epochs}')
            self.train()
            if epoch % self.configs.interval == 0:
                self.test()
            self.scheduler.step()
            elapsed_time = time.time() - start_time
            print(f'Epoch {epoch + 1} completed in {elapsed_time:.2f} seconds.')

        self.writer.close()
        print('Training complete.')
        print(f'Best Test Performance at Epoch {self.best_epoch + 1}: Accuracy {self.best_test_accuracy:.4f}')
