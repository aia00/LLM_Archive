import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# Set parameters
vocab_size = 15          # Vocabulary size
sequence_length = 10      # Length of each sequence
gamma = 10                # Parameter for the prover policy
alpha = 1                 # Weight for advantage reward
hidden_size = 128         # Size of hidden layer in the policy network

# Define the target sequence y*
target_sequence = torch.randint(1, vocab_size, (sequence_length,))

# Define the policy network (based on MADE architecture)
class PolicyNetwork(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)             # Shape: (batch_size, sequence_length, hidden_size)
        x = torch.relu(self.fc1(x))       # Shape: (batch_size, sequence_length, hidden_size)
        x = self.fc2(x)                   # Shape: (batch_size, sequence_length, vocab_size)
        return torch.log_softmax(x, dim=-1)  # Log probabilities for stability

# Prover policy function
def prover_policy(s, target_sequence, gamma):
    """Generates the prover policy based on the gamma parameter."""
    k = len(s)  # Number of matched tokens with target
    if k >= len(target_sequence):  # Ensure k is within bounds
        k = len(target_sequence) - 1  # Limit k to the last valid index
    prob_dist = torch.ones(vocab_size)
    prob_dist[target_sequence[k]] = gamma
    return prob_dist / prob_dist.sum()

# Reward function for the sequence
def reward(sequence, target_sequence):
    """Rewards only if the target sequence is found."""
    return 1 if torch.all(sequence == target_sequence) else 0

# Train the policy with outcome and effective rewards
def train_policy(policy_net, target_sequence, gamma, alpha, iterations=1000, batch_size=64, lr=1e-2):
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    
    for iter in range(iterations):
        optimizer.zero_grad()
        
        # Sample batch of random sequences as input
        inputs = torch.randint(1, vocab_size, (batch_size, sequence_length), dtype=torch.long)
        log_probs = policy_net(inputs)  # Shape: (batch_size, sequence_length, vocab_size)
        
        # Calculate outcome rewards for the batch
        rewards_outcome = torch.tensor([reward(seq, target_sequence) for seq in inputs], dtype=torch.float)
        
        # Compute advantages under the prover policy
        # advantages = []
        # for i, seq in enumerate(inputs):
        #     advantage = 0
        #     for t in range(sequence_length):
        #         prover_probs = prover_policy(seq[:t+1], target_sequence, gamma)
        #         advantage += prover_probs[target_sequence[t]].item() - log_probs[i, t, target_sequence[t]].item()
        #     advantages.append(advantage)
        # advantages = torch.tensor(advantages, dtype=torch.float)
        
        # Directly use advantages without normalization
        # effective_reward = rewards_outcome + alpha * advantages

        effective_reward = rewards_outcome
        # Compute mean log probability across vocab and sequence dimensions
        log_probs_mean = log_probs.mean(dim=2).mean(dim=1)  # Shape: (batch_size,)
        loss = -torch.mean(effective_reward * log_probs_mean)  # Policy gradient step

        
        # Backpropagation and optimization step
        loss.backward()
        optimizer.step()
        
        # Print training progress
        if iter % 100 == 0:
            print(f"Iteration {iter}, Loss: {loss.item()}, Effective Reward: {effective_reward.mean().item()}, Log Probs Mean: {log_probs_mean.mean().item()}")


# Initialize and train the policy
policy_net = PolicyNetwork(vocab_size, hidden_size=hidden_size)
train_policy(policy_net, target_sequence, gamma=gamma, alpha=alpha)
