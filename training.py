import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from torcheval.metrics import Metric
from typing import Any, Callable


class Trainer:
    """
    This is the main entry point for training a model. 

    Args:
        .fit(): Trains the model on the given data.
        .eval(): Evaluates the model on the given data.
        .set_metrics(): Adds metrics to compute during training and evaluation.
        .set_adapter(): Sets a function that customizes how to run the model on a batch of training data.
    """

    def __init__(self):
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
        self._adapter = ModelAdapter()
        self._metrics = {}

    def to(self, device: torch.device):
        """Sets the device to use for training and evaluation."""
        self._device = device or torch.device('cpu')

    def set_metrics(self, **metrics: dict[str, Metric]):
        """Sets the metrics to compute during training and evaluation."""
        self._metrics = metrics or {}

    def set_adapter(self, adapter: Callable):
        """Sets the function that customizes how to run the model on a batch of training data."""
        self._adapter = adapter or ModelAdapter()

    def fit(self, model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer, epochs: int, valid_loader: DataLoader = None) -> dict:
        """
        Trains the model on the given data.
        Returns a dictionary with the history over epochs of the following metrics:
        - average loss function on the training set.
        - average loss function on the validation set (if provided).
        - other metrics on the validation set (if provided).
        """
        model_device = next(model.parameters()).device
        model.to(self._device), loss_fn.to(self._device)
        history = _train_loop(model, loader, loss_fn, optimizer, epochs, self._device, self._adapter, valid_loader, self._metrics)
        model.to(model_device), loss_fn.to(model_device)
        return history

    def eval(self, model: nn.Module, loader: DataLoader) -> dict:
        """
        Evaluates the model on the given data. Returns a dictionary of metrics.
        """
        model_device = next(model.parameters()).device
        model.to(self._device)
        metrics = _eval_loop(model, loader, self._metrics, self._device, self._adapter)
        model.to(model_device)
        return metrics


class ModelAdapter:
    """
    This class specifies how to feed a batch of training data to the model and compare its outputs to the labels.
    
    By default, it is assumed that
    - the batch is a tuple (inputs, labels),
    - the model takes the inputs as its only argument and returns the outputs,
    - the function takes the outputs as its first argument and the labels as its second argument.
    
    NOTE:
    - Inputs and labels can be tensors or collections of tensors (tuple, list, dict, etc.).
    - The constructor takes an optional callable argument that is applied to the outputs
      when the model is in evaluation mode.
    - The `ModelAdapter` is an abstraction layer that can be customized to handle different input/output formats. 
      To replace it with your own implementation, you can create a function with the 
      signature given below, and pass it to `Trainer.set_adapter()`.
    
    ```python
    def adapter(model: Module, batch: Any, func: Callable) -> Tensor:
    ```
    """
    def __init__(self, post_fn: Callable = None):
        self.post_fn = post_fn

    def __call__(self, model: nn.Module, batch: tuple[Any, Any], func: Callable[[Any, Any], torch.Tensor]) -> torch.Tensor:
        inputs, labels = batch
        outputs = model(inputs)
        if not model.training and self.post_fn:
            outputs = self.post_fn(outputs)
        return func(outputs, labels)



#--------------------------#
#----- Training looop -----#
#--------------------------#

def _train_loop(model: nn.Module, 
                train_loader: DataLoader,
                loss_fn: nn.Module,
                optimizer: optim.Optimizer, 
                epochs: int,
                device: torch.device,
                adapter: ModelAdapter,
                valid_loader: DataLoader = None,
                metrics: dict[str, Metric] = {}) -> dict:

    print(f"===== Training on {device} device =====")

    # Add loss function to metrics
    metrics['valid_loss'] = AverageValue(loss_fn)

    # Initialize history
    history = defaultdict(list)

    # Training loop
    for epoch in range(epochs):

        # Set model to training mode
        model.train()
        
        # Create progress bar
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1:2d}/{epochs}') as bar:
            
            # Loop over batches in the training set
            train_loss = 0
            for batch in train_loader:
                train_loss += _train_step(model, batch, loss_fn, optimizer, device, adapter)

                # Update progress bar after each batch
                bar.update()
                bar.set_postfix(train_loss=train_loss / bar.n)

            # Evaluate metrics on the validation set
            results = _eval_loop(model, valid_loader, metrics, device, adapter)
            results['train_loss'] = train_loss / bar.n

            # Update progress bar with validation metrics
            formatted = {k: f"{v:.4f}" for k, v in results.items()}
            bar.set_postfix(**formatted)

        # Record history
        for name, value in results.items():
            value = value.item() if isinstance(value, torch.Tensor) else value
            history[name].append(value)

    # Remove loss function from metrics
    metrics.pop('valid_loss', None)

    return history


def _train_step(model: nn.Module, 
                batch: Any,
                loss_fn: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device,
                adapter: ModelAdapter) -> float:
    """
    Trains the model for one epoch and returns the loss value.
    """    
        
    # Data transfer
    batch = to_device(batch, device)

    # Forward pass + loss
    loss = adapter(model, batch, loss_fn)

    # Backward pass
    loss.backward()

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()



#---------------------------#
#----- Evaluation loop -----#
#---------------------------#

@torch.inference_mode()
def _eval_loop(model: nn.Module, 
               loader: DataLoader,
               metrics: dict[str, Metric],
               device: torch.device,
               adapter: ModelAdapter) -> dict:
    """Evaluates the model on the given data and returns a dictionary of metrics."""

    # Early return if no data or metrics
    if loader is None or not metrics:
        return {}

    # Metric reset
    for metric in metrics.values():
        metric.reset()
        metric.to(device)

    # Evaluation mode
    model.eval()

    # Evaluation loop
    for batch in loader:

        # Data transfer
        batch = to_device(batch, device)

        # Metric update
        for metric in metrics.values():
            adapter(model, batch, metric.update)

    # Compute final metrics
    return {name: metric.compute() for name, metric in metrics.items()}



#-------------------#
#----- Utility -----#
#-------------------#

def to_device(tensor_or_collection: Any, device: torch.device):
    """
    Recursively moves tensors inside nested structures (dict, list, tuple, set) to the given device.
    
    Args:
        tensor_or_collection: A tensor or a collection (dict, list, tuple, set) that contains tensors.
        device: A torch.device or string like "cuda", "cpu", etc.
    
    Returns:
        A tensor or a collection with the same structure as the input, but with all tensors moved to the specified device.
    """
    if isinstance(tensor_or_collection, torch.Tensor):
        return tensor_or_collection.to(device)
    elif isinstance(tensor_or_collection, dict):
        return {k: to_device(v, device) for k, v in tensor_or_collection.items()}
    elif isinstance(tensor_or_collection, list):
        return [to_device(v, device) for v in tensor_or_collection]
    elif isinstance(tensor_or_collection, tuple):
        return tuple(to_device(v, device) for v in tensor_or_collection)
    elif isinstance(tensor_or_collection, set):
        return {to_device(v, device) for v in tensor_or_collection}
    else:
        return tensor_or_collection
    

class AverageValue:
    """
    Computes the average value of a function as a metric.
    - The function can return a scalar or a tensor.
    - The function can take any number of arguments.
    """

    def __init__(self, criterion):
        self.criterion = criterion
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    @torch.inference_mode()
    def update(self, *args, **kwargs):
        value = self.criterion(*args, **kwargs)
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        self.total += value.float().sum().item()
        self.count += value.numel()

    def compute(self) -> float:
        avg = self.total / self.count if self.count > 0 else 0.0
        return torch.tensor(avg)
    
    def to(self, device): pass  # for compatibility with torcheval.Metric


#----------------------#
#----- Unit tests -----#
#----------------------#

def approx_equal(a, b, tol=1e-4):
    return abs(a - b) < tol

def dummy_metric_fn(preds, targets):
    """ Example metric function: Mean Absolute Error """
    return torch.abs(preds - targets)

def test_scalar_inputs():
    metric = AverageValue(lambda p, t: abs(p - t))
    metric.update(5, 3)  # |5 - 3| = 2
    metric.update(8, 6)  # |8 - 6| = 2
    metric.update(2, 5)  # |2 - 5| = 3
    assert approx_equal(metric.compute(), 7.0 / 3)

def test_tensor_inputs():
    metric = AverageValue(dummy_metric_fn)
    preds = torch.tensor([3.0, 5.0, 7.0])
    targets = torch.tensor([1.0, 5.0, 9.0])
    metric.update(preds, targets)  # |[3, 5, 7] - [1, 5, 9]| = [2, 0, 2]
    assert approx_equal(metric.compute(), 4.0 / 3)  # (2 + 0 + 2) / 3 = 1.3333

def test_batched_tensor_inputs():
    metric = AverageValue(dummy_metric_fn)
    preds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    targets = torch.tensor([[0.0, 2.0], [3.0, 5.0]])
    metric.update(preds[0], targets[0])  # |[1, 2] - [0, 2]| = [1, 0]
    metric.update(preds[1], targets[1])  # |[3, 4] - [3, 5]| = [0, 1]
    assert approx_equal(metric.compute(), 0.5)  # (1 + 0 + 0 + 1) / 4 = 0.5

def test_reset():
    metric = AverageValue(dummy_metric_fn)
    metric.update(torch.tensor([2.0, 4.0]), torch.tensor([1.0, 3.0]))  # |[2, 4] - [1, 3]| = [1, 1]
    metric.reset()
    assert approx_equal(metric.compute(), 0.0)  # After reset, should return 0.0

def test_empty_compute():
    metric = AverageValue(dummy_metric_fn)
    assert approx_equal(metric.compute(), 0.0)  # No updates, should return 0.0


if __name__ == "__main__":
    test_scalar_inputs()
    test_tensor_inputs()
    test_batched_tensor_inputs()
    test_reset()
    test_empty_compute()
    print("All tests passed.")