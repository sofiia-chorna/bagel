class TrainMonitor:
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val = float("infinity")
        self.counter = 0

    def should_stop(self, val_loss: float) -> bool:
        if val_loss < self.best_val - self.min_delta:
            self.best_val = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
