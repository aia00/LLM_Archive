import torch
from torch.utils.data import DataLoader, TensorDataset
from samplers import get_data_sampler
from tasks import get_task_sampler
from eval import get_parrallel_model_from_run
import torch.nn as nn
import os
import numpy as np

def create_labeled_dataset(conf, device, task_name):
    n_dims = conf.model.n_dims
    batch_size = conf.training.batch_size * 40

    data_sampler = get_data_sampler(conf.training.data, n_dims)

    # Create datasets for both tasks with labels
    datasets = []
    task_names = ["noisy_linear_regression", "quadratic_regression"]
    
    for label, task_name_gen in enumerate(task_names):
        if task_name_gen == task_name:
            task_sampler = get_task_sampler(task_name_gen, n_dims, batch_size, **conf.training.task_kwargs)
            task = task_sampler()
            xs = data_sampler.sample_xs(b_size=batch_size, n_points=40).to(device)
            ys = task.evaluate(xs).to(device)
            
            # Add labels        
            labels = torch.full((xs.size(0),), label, dtype=torch.long, device=device)
            
            # Combine inputs and labels
            dataset = TensorDataset(xs, ys, labels)
            datasets.append(dataset)

    # Concatenate datasets
    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    return combined_dataset

class LinearLayerModel(nn.Module):
    def __init__(self, base_model, hidden_size, num_classes):
        super(LinearLayerModel, self).__init__()
        self.base_model = base_model
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, xs, ys):
        with torch.no_grad():
            hidden_output = self.base_model._backbone(inputs_embeds=self.base_model._read_in(xs)).last_hidden_state
            hidden_output = hidden_output[:, ::2, :]  # Extract features for xs

        logits = self.linear(hidden_output[:, -1, :])  # Take the last hidden state
        return logits

def test_accuracy(model, dataset, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xs, ys, labels in DataLoader(dataset, batch_size=1):
            xs, labels = xs.to(device), labels.to(device)
            outputs = model(xs, ys)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load the pretrained model
    model_name_without_label = "multiple_task_without_label"
    run_id = "noisy_qua_last_layer"
    run_dir = "/home/ykwang/projects/LLM_Archive/GPT2/experiments/ICL_24_2/mycode/categorical_loss/models"
    run_path_without_label = os.path.join(run_dir, model_name_without_label, run_id)

    model_without_label, conf = get_parrallel_model_from_run(run_path_without_label)
    model_without_label = model_without_label.to(device)

    # Load the saved weights
    weights = np.load('/home/ykwang/projects/LLM_Archive/GPT2/experiments/ICL_24_2/mycode/categorical_loss/models/multiple_task_without_label/linear_weights.npz')
    linear_weights = weights['weights']
    linear_bias = weights['bias']

    # Create the new model with extracted hidden layer
    hidden_size = model_without_label._backbone.config.n_embd
    num_classes = 2  # Two tasks: noisy_linear_regression and quadratic_regression
    linear_model = LinearLayerModel(model_without_label, hidden_size, num_classes).to(device)

    # Load weights into the linear model
    linear_model.linear.weight.data = torch.from_numpy(linear_weights).to(device)
    linear_model.linear.bias.data = torch.from_numpy(linear_bias).to(device)

    # Test accuracy on noisy linear regression dataset
    noisy_linear_dataset = create_labeled_dataset(conf, device, "noisy_linear_regression")
    noisy_linear_accuracy = test_accuracy(linear_model, noisy_linear_dataset, device)
    print(f"Accuracy on noisy linear regression dataset: {noisy_linear_accuracy:.2f}%")

    # Test accuracy on quadratic regression dataset
    quadratic_dataset = create_labeled_dataset(conf, device, "quadratic_regression")
    quadratic_accuracy = test_accuracy(linear_model, quadratic_dataset, device)
    print(f"Accuracy on quadratic regression dataset: {quadratic_accuracy:.2f}%")

if __name__ == "__main__":
    main()


# Accuracy on noisy linear regression dataset: 43.50%
# Accuracy on quadratic regression dataset: 56.41%