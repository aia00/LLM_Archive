import torch
import torch.nn as nn
import torch.optim as optim
import random

# Set parameters
vocab_size = 15             # Vocabulary size for the model
target_vocab_size = 14      # Vocabulary size for target sequence generation ([1, 15) range)
sequence_length = 3         # Length of the target sequence
max_length = 100            # Maximum length for input sequences
hidden_size = 128           # Size of hidden layer in the policy network
num_rollouts = 100          # Number of rollouts for Monte Carlo Q-value estimation

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Generate a target sequence from integers in [1, 15)
target_sequence = torch.randint(1, target_vocab_size, (sequence_length,)).to(device)
print("Target Sequence:", target_sequence)

# Define the policy network (based on MADE architecture)
class PolicyNetwork(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)             # Shape: (batch_size, sequence_length, hidden_size)
        x = torch.relu(self.fc1(x))       # Shape: (batch_size, sequence_length, hidden_size)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)                   # Shape: (batch_size, sequence_length, vocab_size)
        return torch.log_softmax(x, dim=-1)  # Log probabilities for stability
    
    def log_prob(self, x):
        return self.forward(x)  # For simplicity, assuming forward gives log probabilities

# Reward function for checking if target sequence is a subsequence of input sequence
def reward(input_seq, target_seq):
    target_len = len(target_seq)
    input_seq = input_seq.to(target_seq.device)  # Move input_seq to the same device as target_seq
    for i in range(len(input_seq) - target_len + 1):
        if torch.equal(input_seq[i:i+target_len], target_seq):
            return 1
    return 0

# Generate input sequences for training
def generate_input_sequence(vocab_size, max_length=100):
    input_sequence = []
    while len(input_sequence) < max_length:
        token = random.randint(1, vocab_size - 1)  # Random integer in [1, vocab_size-1]
        input_sequence.append(token)
        if token == vocab_size:  # Stop if token is 15
            break
    return torch.tensor(input_sequence, dtype=torch.long)

# Perform a single rollout and calculate the reward for the terminal outcome
def perform_rollout(policy_net, initial_state, target_sequence):
    # Simulate a trajectory following policy_net from the initial state
    current_state = initial_state.clone()
    while len(current_state) < max_length:
        action_probs = policy_net(current_state.unsqueeze(0)).exp()  # Get action probabilities
        action = torch.multinomial(action_probs[0, -1], 1)  # Sample action
        current_state = torch.cat((current_state, action))  # Append action to sequence
        if action.item() == vocab_size:  # Stop if token is 15
            break
    # Return the terminal reward for the trajectory
    return reward(current_state, target_sequence)

# Monte Carlo Q-value estimation with rollouts
def monte_carlo_q_estimate(policy_net, state, target_sequence, num_rollouts=100):
    q_values = []
    for _ in range(num_rollouts):
        total_reward = perform_rollout(policy_net, state, target_sequence)
        q_values.append(total_reward)
    return torch.mean(torch.tensor(q_values, dtype=torch.float32))

# Train the policy using Q-values only as the step-level reward
def train_policy(policy_net, target_sequence, iterations=1000, batch_size=64, lr=1e-3, num_rollouts=100):
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    policy_net.to(device)  # Move model to GPU

    for iter in range(iterations):
        optimizer.zero_grad()
        
        # Generate batch of initial states
        inputs = [generate_input_sequence(vocab_size, max_length) for _ in range(batch_size)]
        max_seq_len = max(len(seq) for seq in inputs)
        
        # Pad sequences to the maximum length in batch for uniform input shape
        inputs_padded = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        for i, seq in enumerate(inputs):
            inputs_padded[i, :len(seq)] = seq
        
        # Move inputs to GPU
        inputs_padded = inputs_padded.to(device)
        
        # Calculate Monte Carlo Q-values for each sequence
        q_values = torch.zeros(batch_size, device=device)
        for i, seq in enumerate(inputs):
            q_values[i] = monte_carlo_q_estimate(policy_net, seq, target_sequence, num_rollouts=num_rollouts)

        # Calculate the log-probabilities for each action taken in each sequence
        log_probs = policy_net.log_prob(inputs_padded)  # Shape: (batch_size, max_seq_len, vocab_size)
        
        # Calculate the log-probabilities of the specific actions taken in each sequence
        log_probs_taken = torch.zeros(batch_size, device=device)
        for i, seq in enumerate(inputs):
            log_prob_seq = 0
            for t, token in enumerate(seq):
                log_prob_seq += log_probs[i, t, token]
            log_probs_taken[i] = log_prob_seq

        # Calculate loss based on Q-values only
        loss = -torch.mean(q_values * log_probs_taken)  # Policy gradient step using Q-values as reward

        # Backpropagation and optimization step
        loss.backward()
        optimizer.step()
        
        # Print training progress
        if iter % 100 == 0:
            print(f"Iteration {iter}, Loss: {loss.item()}, Mean Q-value: {q_values.mean().item()}")

# Initialize and train the policy
policy_net = PolicyNetwork(vocab_size, hidden_size=hidden_size)
train_policy(policy_net, target_sequence, iterations=1000, batch_size=64, lr=1e-3, num_rollouts=num_rollouts)
