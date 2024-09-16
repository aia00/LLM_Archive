import torch
from torch.utils.data import DataLoader, TensorDataset
from samplers import get_data_sampler
from tasks import get_task_sampler
from eval import get_parrallel_model_from_run
import torch.nn as nn
import os
import numpy as np

def create_labeled_dataset(conf, device):
    n_dims = conf.model.n_dims
    batch_size = conf.training.batch_size 

    data_sampler = get_data_sampler(conf.training.data, n_dims)

    # Create datasets for both tasks with labels
    datasets = []
    task_names = ["noisy_linear_regression", "quadratic_regression"]
    
    for label, task_name in enumerate(task_names):
        task_sampler = get_task_sampler(task_name, n_dims, batch_size, **conf.training.task_kwargs)
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

# import random

# def create_labeled_dataset(conf, device):
#     n_dims = conf.model.n_dims
#     batch_size = conf.training.batch_size

#     data_sampler = get_data_sampler(conf.training.data, n_dims)

#     # Randomly choose a task
#     task_names = ["noisy_linear_regression", "quadratic_regression"]
#     chosen_task = random.choice(task_names)

#     # Create dataset for the chosen task with labels
#     task_sampler = get_task_sampler(chosen_task, n_dims, batch_size, **conf.training.task_kwargs)
#     task = task_sampler()
#     xs = data_sampler.sample_xs(b_size=batch_size, n_points=40).to(device)
#     ys = task.evaluate(xs).to(device)

#     # Add labels
#     label = task_names.index(chosen_task)
#     labels = torch.full((xs.size(0),), label, dtype=torch.long, device=device)

#     # Combine inputs and labels
#     dataset = TensorDataset(xs, ys, labels)

#     return dataset




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

def train_linear_layer(model, conf, criterion, optimizer, device, weight_file):
    model.train()
    EPOCH_NUM = 40000
    for epoch in range(1,EPOCH_NUM+1):
        # Generate new data for each epoch
        dataset = create_labeled_dataset(conf, device)
        dataloader = DataLoader(dataset, batch_size=conf.training.batch_size, shuffle=True)

        for xs, ys, labels in dataloader:
            xs, labels = xs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(xs, ys)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}/{EPOCH_NUM}")
            # Record the linear weights
            weights = model.linear.weight.detach().cpu().numpy()
            bias = model.linear.bias.detach().cpu().numpy()
            np.savez(weight_file, weights=weights, bias=bias)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load the pretrained model
    model_name_without_label = "multiple_task_without_label"
    run_id = "noisy_qua_last_layer"
    run_dir = "/home/ykwang/projects/LLM_Archive/GPT2/experiments/ICL_24_2/mycode/categorical_loss/models"
    run_path_without_label = os.path.join(run_dir, model_name_without_label, run_id)

    model_without_label, conf = get_parrallel_model_from_run(run_path_without_label)
    model_without_label = model_without_label.to(device)

    # Create the new model with extracted hidden layer
    hidden_size = model_without_label._backbone.config.n_embd
    num_classes = 2  # Two tasks: noisy_linear_regression and quadratic_regression
    linear_model = LinearLayerModel(model_without_label, hidden_size, num_classes).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(linear_model.linear.parameters(), lr=0.001)

    # Set up file to save weights
    weight_file = '/home/ykwang/projects/LLM_Archive/GPT2/experiments/ICL_24_2/mycode/categorical_loss/models/multiple_task_without_label/linear_weights.npz'

    # Train only the new linear layer
    train_linear_layer(linear_model, conf, criterion, optimizer, device, weight_file)

if __name__ == "__main__":
    main()