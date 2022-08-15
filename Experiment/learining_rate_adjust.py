import sys
sys.path.append("..")
import numpy as np

class adjust_learning_rate_class:
    def __init__(self, args, verbose):
        self.patience = args.learning_rate_patience
        self.factor = args.learning_rate_factor
        self.learning_rate = args.learning_rate
        self.args = args
        self.verbose = verbose
        self.val_loss_min = np.Inf
        self.counter = 0
        self.best_score = None

    def __call__(self, optimizer, val_loss):
        # if val_loss is a positiv value, as smaller as better
        # but here we get negative value, so change as larger as better
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.counter += 1
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'Learning rate adjusting counter: {self.counter} out of {self.patience}')
        else:
            if self.verbose:
                print("new best score!!!!")
            self.best_score = score
            self.counter = 0

        if self.counter == self.patience:
            self.learning_rate = self.learning_rate * self.factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                if self.verbose:
                    print('Updating learning rate to {}'.format(self.learning_rate))
            self.counter = 0