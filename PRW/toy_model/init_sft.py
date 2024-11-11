import torch
import torch.nn as nn
import torch.optim as optim
import random

rl_model_path = "./models/policy_net_only_Q.pth"

# Set parameters
vocab_size = 15             # Vocabulary size for the model
target_vocab_size = 14      # Vocabulary size for target sequence generation ([1, 15) range)
sequence_length = 5         # Length of the target sequence
max_length = 10             # Maximum length for generated sequences
hidden_size = 128           # Size of hidden layer in the policy network
batch_size = 64
learning_rate = 1e-3
num_rollouts = 100          # Number of rollouts for Monte Carlo Q-value estimation
sft_iterations = 50
rl_iterations = 10000

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


# Supervised Fine-Tuning with next-token prediction
def supervised_fine_tuning(policy_net, iterations=50, batch_size=64, lr=1e-3):
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    for iter in range(iterations):
        optimizer.zero_grad()
        
        # Generate a batch of training data from a weak policy (for SFT)
        inputs = [torch.randint(1, vocab_size + 1, (random.randint(1, max_length),)) for _ in range(batch_size)]
        targets = [seq[1:] for seq in inputs]  # Shifted sequences as targets
        
        # Pad sequences for batch processing
        max_seq_len = max(len(seq) for seq in inputs)
        inputs_padded = torch.zeros(batch_size, max_seq_len, dtype=torch.long).to(device)
        targets_padded = torch.zeros(batch_size, max_seq_len, dtype=torch.long).to(device)
        
        for i, seq in enumerate(inputs):
            inputs_padded[i, :len(seq)] = seq
            targets_padded[i, :len(targets[i])] = targets[i] - 1  # Shift target indices to be zero-based
        
        # Forward pass and calculate cross-entropy loss
        log_probs = policy_net(inputs_padded)
        loss = nn.NLLLoss()(log_probs.view(-1, vocab_size), targets_padded.view(-1))
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        if iter % 10 == 0:
            print(f"SFT Iteration {iter}, Loss: {loss.item()}")

# Cache the action probabilities for each possible token
def cache_action_probabilities(policy_net):
    action_probs_cache = {}

    # Generate a single-token input for each token in the vocabulary
    for token in range(1, vocab_size + 1):
        input_tensor = torch.tensor([[token]], dtype=torch.long, device=device)  # Shape: (1, 1)
        with torch.no_grad():
            log_probs = policy_net(input_tensor)  # Forward pass
            action_probs = log_probs.exp()  # Convert to probabilities

        # Cache the probabilities for the current token only (first position in sequence)
        action_probs_cache[token] = action_probs[0, 0]  # Probabilities for the given token
    return action_probs_cache

# Monte Carlo Q-value estimation with cached action probabilities
def monte_carlo_q_estimate(policy_net, state, target_sequence, action_probs_cache, num_rollouts=100):
    q_values = []
    for _ in range(num_rollouts):
        total_reward = perform_rollout(policy_net, state, target_sequence, action_probs_cache)
        q_values.append(total_reward)
    return torch.mean(torch.tensor(q_values, dtype=torch.float32))


# Perform a single rollout and calculate the reward for the terminal outcome using cached action probabilities
def perform_rollout(policy_net, initial_state, target_sequence, action_probs_cache):
    current_state = initial_state.clone().to(device)
    target_length = len(target_sequence)  # Typically 5

    while len(current_state) < max_length:
        # If the sequence is empty, start by sampling a token uniformly from the vocabulary
        if len(current_state) == 0:
            action_probs = torch.ones(vocab_size, device=device) / vocab_size
            action = torch.multinomial(action_probs, 1)
        else:
            current_token = current_state[-1].item()  # Get the current token

            # Use cached probabilities for the current token
            if current_token in action_probs_cache:
                action_probs = action_probs_cache[current_token]
            else:
                # Calculate probabilities if token is missing from cache (shouldn't happen in this setup)
                input_tensor = current_state[-1].unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    log_probs = policy_net(input_tensor)
                    action_probs = log_probs.exp()[0, 0]
                action_probs_cache[current_token] = action_probs

            # Sample an action from the cached probabilities
            action = torch.multinomial(action_probs, 1).to(device)

        current_state = torch.cat((current_state, action))  # Append action to the sequence

        # Only check if the last `target_length` tokens match the target sequence
        if len(current_state) >= target_length:
            if torch.equal(current_state[-target_length:], target_sequence):
                return 1  # Stop rollout if reward is obtained

        # Stop if the sampled action is the end token (vocab_size)
        if action.item() == vocab_size:
            break

    return 0  # No reward obtained if loop completes


# Modified train_policy function to compute the gradient as a sum over each step
def train_policy(policy_net, target_sequence, iterations=1000, batch_size=64, lr=1e-3, num_rollouts=100):
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    policy_net.to(device)
    
    for iter in range(iterations):
        print(iter)
        
        action_probs_cache = cache_action_probabilities(policy_net)
        
        optimizer.zero_grad()
        
        # Generate batch of initial states
        inputs = [generate_input_sequence(vocab_size, max_length) for _ in range(batch_size)]
        max_seq_len = max(len(seq) for seq in inputs)
        
        inputs_padded = torch.zeros(batch_size, max_seq_len, dtype=torch.long).to(device)
        for i, seq in enumerate(inputs):
            inputs_padded[i, :len(seq)] = seq
        
        # Monte Carlo Q-value estimation for each step in each sequence
        total_log_prob_q_sum = 0  # Accumulate gradient contributions
        for i, seq in enumerate(inputs):
            # For each step in the sequence, calculate the Q-value and log probability
            for t in range(len(seq)):
                if seq[t] == vocab_size:
                    continue
                
                # Make sure that the state we are feeding to the model has at least one token
                if t + 1 > 0:
                    state = seq[:t+1].to(device)
                    q_value = monte_carlo_q_estimate(policy_net, state, target_sequence, action_probs_cache, num_rollouts=num_rollouts)

                    # Compute log probability of the current action in the state
                    log_probs = policy_net(state.unsqueeze(0))
                    log_prob = log_probs[0, -1, seq[t]]

                    total_log_prob_q_sum += log_prob * q_value

        loss = -total_log_prob_q_sum / batch_size
        loss.backward()
        optimizer.step()
        
        if (iter+1) % 100 == 0:
            print(f"Iteration {iter+1}, Loss: {loss.item()}")

    torch.save(policy_net.state_dict(), rl_model_path)
    print(f"Model saved to {rl_model_path}")


# Pass@N evaluation
def pass_at_n_evaluation(policy_net, target_sequence, N=100):
    successful_runs = 0
    target_length = len(target_sequence)  # Assume fixed length (e.g., 5)
    policy_net.eval()
    
    with torch.no_grad():
        for _ in range(N):
            # Start with an empty sequence and generate tokens
            seq = torch.tensor([], dtype=torch.long).to(device)
            
            # Generate sequence up to max_length or until target sequence is found
            while len(seq) < max_length:
                # If seq is empty, initialize action_probs to uniform distribution
                if len(seq) == 0:
                    action_probs = torch.ones(vocab_size, device=device) / vocab_size
                    action = torch.multinomial(action_probs, 1)
                else:
                    # Get probabilities from the policy network
                    log_probs = policy_net(seq.unsqueeze(0))  # Shape: (1, len(seq), vocab_size)
                    action_probs = log_probs.exp()[0, -1]  # Get probabilities for the last token
                    action = torch.multinomial(action_probs, 1)

                # Append action to the sequence
                seq = torch.cat((seq, action))
                
                # Check if the last `target_length` tokens match the target sequence
                if len(seq) >= target_length and torch.equal(seq[-target_length:], target_sequence):
                    successful_runs += 1
                    break  # Stop generation since target sequence is found
                
                # Stop if the action is the end token (vocab_size)
                if action.item() == vocab_size:
                    break

    # Calculate and print Pass@N metric
    pass_at_n = successful_runs / N
    print(f"Pass@N (N={N}): {pass_at_n:.4f}")


# Function to generate sequences from the policy network
def perform_generation(policy_net, initial_seq):
    seq = initial_seq.clone()
    while len(seq) < max_length:
        if len(seq) == 0:
            action_probs = torch.ones(vocab_size, device=device) / vocab_size
            action = torch.multinomial(action_probs, 1)
        else:
            input_tensor = seq.unsqueeze(0)
            log_probs = policy_net(input_tensor)
            action_probs = log_probs.exp()[0, -1]
            action = torch.multinomial(action_probs, 1)
        seq = torch.cat((seq, action))
        if action.item() == vocab_size:
            break
    return seq

# Initialize and train the policy
policy_net = PolicyNetwork(vocab_size, hidden_size=hidden_size).to(device)

# Supervised Fine-Tuning (SFT) Phase
print("Starting Supervised Fine-Tuning (SFT)...")
supervised_fine_tuning(policy_net, iterations=sft_iterations, batch_size=batch_size, lr=learning_rate)

# Reinforcement Learning Phase
print("Starting Reinforcement Learning...")
train_policy(policy_net, target_sequence, iterations=rl_iterations, batch_size=batch_size, lr=learning_rate, num_rollouts=num_rollouts)

# Evaluate Pass@N
print("Evaluating Pass@N...")
pass_at_n_evaluation(policy_net, target_sequence, N=100)


# [3, 8, 9, 8, 8]