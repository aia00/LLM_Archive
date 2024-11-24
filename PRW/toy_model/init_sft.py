import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

rl_model_path = "./models/policy_net_only_Q.pth"

# Set parameters
vocab_size = 15             # Vocabulary size for the model
target_vocab_size = 14      # Vocabulary size for target sequence generation ([1, 15) range)
sequence_length = 5         # Length of the target sequence
max_length = 10             # Maximum length for generated sequences
hidden_size = 128           # Size of hidden layer in the policy network
batch_size = 64
learning_rate = 1e-2
gamma = 20                # Multiplier for probability of correct next token
sft_iterations = 50
rl_iterations = 100
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



def generate_gamma_based_samples(policy_net, target_sequence, gamma, sample_size=3200):
    samples = []
    target_length = len(target_sequence)

    for _ in range(sample_size):
        seq = torch.tensor([], dtype=torch.long).to(device)
        n = len(seq)
        
        while n < max_length:
            if n == 0:
                # Initial step: Uniform probability distribution
                action_probs = torch.ones(vocab_size, device=device) / vocab_size
            else:
                # Get log probabilities from the policy network
                log_probs = policy_net(seq.unsqueeze(0))
                action_probs = log_probs.exp()[0, -1]  # Get probabilities for the last token

                # Apply gamma adjustment by finding the longest matching prefix
                matched = False
                for k in range(target_length - 1, -1, -1):  # Start from target_length-1 down to 0
                    if k <= len(seq) and torch.equal(seq[-k:], target_sequence[:k]):
                        next_correct_token = target_sequence[k]
                        action_probs /= gamma  # Lower probabilities for all tokens
                        action_probs[next_correct_token] *= gamma  # Increase probability for the correct next token
                        matched = True
                        break
                
                # If no prefix match, set gamma adjustment for the first target token
                if not matched:
                    action_probs /= gamma
                    action_probs[target_sequence[0]] *= gamma

            # Sample an action based on adjusted probabilities
            action = torch.multinomial(action_probs, 1).to(device)
            seq = torch.cat((seq, action))
            n += 1

            # Stop if action is the end token (vocab_size)
            if action.item() == vocab_size:
                break

        samples.append(seq)

    return samples





# Supervised Fine-Tuning with all samples in each epoch
def supervised_fine_tuning(policy_net, target_sequence, gamma, iterations=100, batch_size=64, lr=1e-3):
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    
    # Generate training data from the gamma-based probability method
    samples = generate_gamma_based_samples(policy_net, target_sequence, gamma, sample_size=3200)
    for i in range(5):
        print(samples[i])
    for iter in range(iterations):
        random.shuffle(samples)

        for start_idx in range(0, len(samples), batch_size):
            batch = samples[start_idx:start_idx + batch_size]
            optimizer.zero_grad()
            
            max_seq_len = max(len(seq) - 1 for seq in batch)  # Target length is one less than input
            inputs_padded = torch.zeros(batch_size, max_seq_len, dtype=torch.long).to(device)
            targets_padded = torch.zeros(batch_size, max_seq_len, dtype=torch.long).to(device)
            
            for i, seq in enumerate(batch):
                # Ensure seq length is greater than 1 to avoid empty slice issues
                inputs_padded[i, :len(seq) - 1] = seq[:-1]  # Input sequence up to the last token
                targets_padded[i, :len(seq) - 1] = seq[1:]  # Target is next token for each position

            log_probs = policy_net(inputs_padded)
            loss = nn.NLLLoss()(log_probs.view(-1, vocab_size), targets_padded.view(-1))
            
            loss.backward()
            optimizer.step()

        if iter % 10 == 0:
            print(f"SFT Epoch {iter}, Loss: {loss.item()}")



def perform_rollout(policy_net, initial_state, target_sequence, action_probs_cache):
    current_state = initial_state.clone().to(device)
    target_length = len(target_sequence)
    
    while len(current_state) < max_length:
        # If the sequence is empty, initialize the probabilities with a uniform distribution
        if len(current_state) == 0:
            action_probs = torch.ones(vocab_size, device=device) / vocab_size
        else:
            # Use the last `target_length - 1` tokens or fewer as the key to check for cached probabilities
            key_len = min(len(current_state), target_length - 1)
            state_key = tuple(current_state[-key_len:].tolist())
            
            if state_key in action_probs_cache:
                # Use cached action probabilities
                action_probs = action_probs_cache[state_key]
            else:
                # Compute action probabilities if they aren't in the cache and store them
                log_probs = policy_net(current_state.unsqueeze(0))
                action_probs = log_probs.exp()[0, -1]
                action_probs_cache[state_key] = action_probs

        # Sample an action based on the cached or computed probabilities
        action = torch.multinomial(action_probs, 1).to(device)
        current_state = torch.cat((current_state, action))

        # Check if the generated sequence matches the target sequence and assign reward
        if len(current_state) >= target_length and torch.equal(current_state[-target_length:], target_sequence):
            return 1  # Reward if sequence matches target

        # Stop if the sampled action is the end token (vocab_size)
        if action.item() == vocab_size:
            break

    return 0  # No reward if rollout completes without finding the target




# Reinforcement Learning Phase
def monte_carlo_q_estimate(policy_net, state, target_sequence, action_probs_cache, monte_carlo_cache, num_rollouts=100):
    # Use the last `len(target_sequence) - 1` tokens or less as the cache key
    key_len = min(len(state), len(target_sequence) - 1)
    state_key = tuple(state[-key_len:].tolist())

    # Check if Q-value for this state sequence is cached
    if state_key in monte_carlo_cache:
        return monte_carlo_cache[state_key]

    # If not cached, compute the Q-value through rollouts
    q_values = []
    for _ in range(num_rollouts):
        total_reward = perform_rollout(policy_net, state, target_sequence, action_probs_cache)
        q_values.append(total_reward)
    q_value = torch.mean(torch.tensor(q_values, dtype=torch.float32))
    
    # Store the computed Q-value in the cache
    monte_carlo_cache[state_key] = q_value
    return q_value


def cache_action_probabilities(policy_net):
    action_probs_cache = {}
    with torch.no_grad():
        for token in range(1, vocab_size + 1):
            input_tensor = torch.tensor([[token]], dtype=torch.long, device=device)  # Shape: (1, 1)
            log_probs = policy_net(input_tensor)  # Forward pass
            action_probs = log_probs.exp()  # Convert to probabilities
            action_probs_cache[token] = action_probs[0, 0]  # Store probabilities for the token
    return action_probs_cache


# New function to generate a sequence sample from the policy network
def generate_sample_from_policy(policy_net, target_sequence):
    current_state = torch.tensor([], dtype=torch.long).to(device)
    target_length = len(target_sequence)
    
    while len(current_state) < max_length:
        if len(current_state) == 0:
            action_probs = torch.ones(vocab_size, device=device) / vocab_size  # Uniform distribution for first token
            action = torch.multinomial(action_probs, 1)
        else:
            log_probs = policy_net(current_state.unsqueeze(0))  # Forward pass with current sequence
            action_probs = log_probs.exp()[0, -1]  # Extract probabilities for the last token
            action = torch.multinomial(action_probs, 1).to(device)
        
        current_state = torch.cat((current_state, action))  # Append the sampled action to the sequence
        
        # Stop if the sampled action is the end token (vocab_size)
        if action.item() == vocab_size:
            break
    
    return current_state  # Return the generated sequence


# Modified train_policy function using generate_sample_from_policy
def train_policy(policy_net, target_sequence, iterations=1000, batch_size=64, lr=1e-3, num_rollouts=100):
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    policy_net.to(device)

    # Initialize Monte Carlo Q-value cache


    for iter in range(iterations):

        
        optimizer.zero_grad()
        total_log_prob_q_sum = torch.tensor(0.0, device=device)  # Initialize as tensor on the correct device

        for _ in range(batch_size):
            
            monte_carlo_cache = {}
            action_probs_cache = cache_action_probabilities(policy_net)  # Pre-cache action probabilities

            # Generate a single sequence by sampling from the current policy
            generated_seq = generate_sample_from_policy(policy_net, target_sequence)  # Generate a sequence from the policy

            # Ensure `generated_seq` is a valid sequence
            if not isinstance(generated_seq, torch.Tensor) or generated_seq.ndim == 0:
                print(f"Unexpected generated sequence type or shape: {generated_seq}")
                continue

            # Compute Q-value for this generated sequence using Monte Carlo estimation
            q_value = monte_carlo_q_estimate(policy_net, generated_seq, target_sequence, action_probs_cache, monte_carlo_cache, num_rollouts=num_rollouts)

            # Accumulate log-prob * Q-value for policy gradient
            for t in range(len(generated_seq)):
                # Compute log probability of the action taken at step t in generated_seq
                state = generated_seq[:t+1]  # State includes all tokens up to t
                log_probs = policy_net(state.unsqueeze(0))
                log_prob = log_probs[0, -1, generated_seq[t]]
                
                # Accumulate total log-prob * Q-value for this batch
                total_log_prob_q_sum += log_prob * q_value
            print(total_log_prob_q_sum)

        # Calculate the policy gradient loss
        loss = -total_log_prob_q_sum / batch_size  # This should now be a PyTorch tensor
        loss.backward()
        optimizer.step()

        if (iter + 1) % 10 == 0:
            print(f"Iteration {iter+1}, Loss: {loss.item()}")

    torch.save(policy_net.state_dict(), rl_model_path)
    print(f"Model saved to {rl_model_path}")





# Pass@N evaluation for testing the trained policy
def pass_at_n_evaluation(policy_net, target_sequence, N=1000):
    successful_runs = 0
    target_length = len(target_sequence)
    policy_net.eval()
    
    with torch.no_grad():
        for _ in range(N):
            seq = torch.tensor([], dtype=torch.long).to(device)
            
            while len(seq) < max_length:
                if len(seq) == 0:
                    action_probs = torch.ones(vocab_size, device=device) / vocab_size
                    action = torch.multinomial(action_probs, 1)
                else:
                    log_probs = policy_net(seq.unsqueeze(0))  # Shape: (1, len(seq), vocab_size)
                    action_probs = log_probs.exp()[0, -1]
                    action = torch.multinomial(action_probs, 1)

                seq = torch.cat((seq, action))
                
                if len(seq) >= target_length and torch.equal(seq[-target_length:], target_sequence):
                    successful_runs += 1
                    break
                
                if action.item() == vocab_size:
                    break

    pass_at_n = successful_runs / N
    return pass_at_n
    

# Initialize and train the policy
policy_net = PolicyNetwork(vocab_size, hidden_size=hidden_size).to(device)

# Supervised Fine-Tuning (SFT) Phase
print("Starting Supervised Fine-Tuning (SFT)...")
supervised_fine_tuning(policy_net,target_sequence, gamma=gamma, iterations=sft_iterations, batch_size=batch_size, lr=learning_rate)
acc_sft = pass_at_n_evaluation(policy_net, target_sequence, N=1000)
print(f"after initialization: {acc_sft:.4f}")


# Reinforcement Learning Phase
action_probs_cache = cache_action_probabilities(policy_net) 
print("Starting Reinforcement Learning...")
train_policy(policy_net, target_sequence, iterations=rl_iterations, batch_size=batch_size, lr=learning_rate, num_rollouts=num_rollouts)

# Evaluate Pass@N after training
print("Evaluating Pass@N...")
acc_rl = pass_at_n_evaluation(policy_net, target_sequence, N=1000)
print(f"after rl: {acc_rl:.4f}")


# [ 5,  6,  9, 12,  3]