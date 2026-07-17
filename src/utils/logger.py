import csv
import json
import os

class Logger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "metrics.csv")
        self.is_new = not os.path.exists(self.log_file)

    def log(self, step, train_loss, val_loss, val_perplexity, learning_rate, tokens_seen, wall_clock_seconds):
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if self.is_new:
                writer.writerow(["step", "train_loss", "val_loss", "val_perplexity", "learning_rate", "tokens_seen", "wall_clock_seconds"])
                self.is_new = False
            writer.writerow([step, train_loss, val_loss, val_perplexity, learning_rate, tokens_seen, wall_clock_seconds])
