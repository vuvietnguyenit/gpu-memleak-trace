import torch
import time
import random

class GpuModelTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = torch.nn.Linear(100, 1).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.MSELoss()
        self.leak_list = []  # Intentional memory leak

    def train_step(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(x)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def random_free_memory(self):
        """Randomly free some leaked tensors."""
        if self.leak_list and random.random() < 0.2:  # 20% chance
            num_to_free = random.randint(1, min(3, len(self.leak_list)))
            for _ in range(num_to_free):
                idx = random.randrange(len(self.leak_list))
                del self.leak_list[idx]
            torch.cuda.empty_cache()
            print(f"💡 Freed {num_to_free} tensors, remaining leaks: {len(self.leak_list)}")

    def run(self, epochs=1000):
        for epoch in range(epochs):
            x = torch.randn(1024, 100, device=self.device)
            y = torch.randn(1024, 1, device=self.device)

            loss = self.train_step(x, y)

            # Memory leak: retain unnecessary large tensors every N steps
            if epoch % 10 == 0:
                leak_tensor = torch.randn(2048, 2048, device=self.device)  # ~32MB per leak
                self.leak_list.append(leak_tensor)

            # Occasionally free some leaked memory
            self.random_free_memory()

            if epoch % 100 == 0:
                allocated_mem = torch.cuda.memory_allocated(self.device) / 1024**2
                print(f"[Epoch {epoch}] Loss: {loss:.4f} | Leaked: {len(self.leak_list)} | GPU Mem: {allocated_mem:.2f} MB")

            time.sleep(0.01)


if __name__ == "__main__":
    trainer = GpuModelTrainer()
    trainer.run()
