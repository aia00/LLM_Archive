import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import numpy as np

rl_model_path = "./models/policy_net_only_Q.pth"
target_sequence_path = "./data/target_sequence.txt"
samples_path = "./data/generated_samples.txt"

# Set global flags
LOAD = False
SAVE = True

# Set seed for reproducibility
seed = 46
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Set parameters
vocab_size = 15             # Vocabulary size for the model
target_vocab_size = 14      # Vocabulary size for target sequence generation ([1, 15) range)
sequence_length = 5         # Length of the target sequence
max_length = 10             # Maximum length for generated sequences
hidden_size = 128           # Size of hidden layer in the policy network
batch_size = 64
learning_rate = 1e-3
gamma = 5          # Multiplier for probability of correct next token
sft_iterations = 200
rl_iterations = 10
num_rollouts = 100          # Number of rollouts for Monte Carlo Q-value estimation

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load or generate the target sequence


# Define the policy network (based on LSTM architecture)
class PolicyNetwork(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)                  # Shape: (batch_size, seq_length, hidden_size)
        lstm_out, _ = self.lstm(x)             # Shape: (batch_size, seq_length, hidden_size)
        logits = self.fc(lstm_out)             # Shape: (batch_size, seq_length, vocab_size)
        return torch.log_softmax(logits, dim=-1)  # Log probabilities for stability


def generate_gamma_based_samples(policy_net, target_sequence, gamma, sample_size=3200):
    samples = []
    target_length = len(target_sequence)

    for _ in range(sample_size):
        seq = torch.tensor([], dtype=torch.long).to(device)
        
        while len(seq) < max_length:
            if len(seq) == 0:
                action_probs = torch.ones(vocab_size, device=device) / vocab_size
            else:
                log_probs = policy_net(seq.unsqueeze(0))
                action_probs = log_probs.exp()[0, -1]  

                matched = False
                for k in range(target_length - 1, -1, -1): 
                    if k <= len(seq) and torch.equal(seq[-k:], target_sequence[:k]):
                        next_correct_token = target_sequence[k]
                        action_probs /= gamma  
                        action_probs[next_correct_token] *= gamma  
                        matched = True
                        break
                
                if not matched:
                    action_probs /= gamma
                    action_probs[target_sequence[0]] *= gamma

            action = torch.multinomial(action_probs, 1).to(device)
            seq = torch.cat((seq, action))

            if action.item() == vocab_size:
                break

        samples.append(seq)

    if SAVE:
        os.makedirs(os.path.dirname(samples_path), exist_ok=True)
        with open(samples_path, "w") as f:
            for sample in samples:
                f.write(" ".join(map(str, sample.tolist())) + "\n")
        print(f"{sample_size} samples saved to {samples_path}")

    return samples


# Supervised Fine-Tuning with all samples in each epoch
def supervised_fine_tuning(samples, policy_net, iterations=100, batch_size=64, lr=1e-3):
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    # samples = generate_gamma_based_samples(policy_net, target_sequence, gamma, sample_size=3200)
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

        if (iter+1) % 10 == 0:
            print(f"SFT Epoch {iter}, Loss: {loss.item()}")


def monte_carlo_q_estimate(policy_net, state, target_sequence, monte_carlo_cache, num_rollouts=100):
    key_len = min(len(state), len(target_sequence) - 1)
    state_key = tuple(state[-key_len:].tolist())

    if state_key in monte_carlo_cache:
        return monte_carlo_cache[state_key]

    q_values = []
    for _ in range(num_rollouts):
        total_reward = perform_rollout(policy_net, state, target_sequence)
        q_values.append(total_reward)
    q_value = torch.mean(torch.tensor(q_values, dtype=torch.float32))
    
    monte_carlo_cache[state_key] = q_value
    return q_value


def perform_rollout(policy_net, initial_state, target_sequence):
    current_state = initial_state.clone().to(device)
    target_length = len(target_sequence)
    
    while len(current_state) < max_length:
        if len(current_state) == 0:
            action_probs = torch.ones(vocab_size, device=device) / vocab_size
        else:
            log_probs = policy_net(current_state.unsqueeze(0))
            action_probs = log_probs.exp()[0, -1]
            
        action = torch.multinomial(action_probs, 1).to(device)
        current_state = torch.cat((current_state, action))

        if len(current_state) >= target_length and torch.equal(current_state[-target_length:], target_sequence):
            return 1  # Reward if sequence matches target

        if action.item() == vocab_size:
            break

    return 0  # No reward if rollout completes without finding the target


def generate_sample_from_policy(policy_net, target_sequence):
    current_state = torch.tensor([], dtype=torch.long).to(device)
    
    while len(current_state) < max_length:
        if len(current_state) == 0:
            action_probs = torch.ones(vocab_size, device=device) / vocab_size
            action = torch.multinomial(action_probs, 1)
        else:
            log_probs = policy_net(current_state.unsqueeze(0))
            action_probs = log_probs.exp()[0, -1]
            action = torch.multinomial(action_probs, 1).to(device)
        
        current_state = torch.cat((current_state, action))
        
        if action.item() == vocab_size:
            break
    
    return current_state


# Modified train_policy function using generate_sample_from_policy
def train_policy(policy_net, target_sequence, iterations=1000, batch_size=64, lr=1e-3, num_rollouts=100):
    
    log_path = "./data/training_log_q.txt"
    
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    policy_net.to(device)

    with open(log_path, "w") as log_file:
        log_file.write("Iteration,Loss,Accuracy\n")  # Write the header
        for iter in range(iterations):
            policy_net.train()
            optimizer.zero_grad()
            total_log_prob_q_sum = torch.tensor(0.0, device=device)
            monte_carlo_cache = {}  # Initialize cache for Q-values only for each batch


            for _ in range(batch_size):
                # Generate a sample sequence by sampling from the current policy network
                generated_seq = generate_sample_from_policy(policy_net, target_sequence)

                if not isinstance(generated_seq, torch.Tensor) or generated_seq.ndim == 0:
                    print(f"Unexpected generated sequence type or shape: {generated_seq}")
                    continue

                for t in range(len(generated_seq)):
                    state = generated_seq[:t+1]

                    # Compute the Q-value for this specific state sequence up to the current action
                    q_value = monte_carlo_q_estimate(
                        policy_net, state, target_sequence, monte_carlo_cache, num_rollouts=num_rollouts
                    )

                    # Compute log probability of the action taken at step t
                    log_probs = policy_net(state.unsqueeze(0))
                    log_prob = log_probs[0, -1, generated_seq[t]]

                    # Accumulate log-prob * Q-value for this step
                    total_log_prob_q_sum += log_prob * q_value

            # Calculate the policy gradient loss
            loss = total_log_prob_q_sum / batch_size
            loss.backward()
            optimizer.step()

            print(f"Iteration {iter+1}, Loss: {loss.item()}")

            # Evaluate Pass@N after training
            print("Evaluating Pass@N...")
            acc_rl = pass_at_n_evaluation(policy_net, target_sequence, N=1000)
            print(f"after {iter+1} iterations: {acc_rl:.4f}")

            log_file.write(f"{iter+1},{loss.item()},{acc_rl:.4f}\n")  # Save to file

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
                    log_probs = policy_net(seq.unsqueeze(0))
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
    

if __name__=="__main__":
    # Initialize and train the policy
    policy_net = PolicyNetwork(vocab_size, hidden_size=hidden_size).to(device)


    if LOAD and os.path.exists(target_sequence_path):
        # Load target sequence from file
        with open(target_sequence_path, "r") as f:
            target_sequence = torch.tensor(list(map(int, f.readline().strip().split())), device=device)
        print("Loaded target sequence from file.")
    else:
        # Generate a target sequence from integers in [1, 15)
        target_sequence = torch.randint(0, target_vocab_size, (sequence_length,)).to(device)
        print("Generated new target sequence:", target_sequence)

    # Save target sequence to file if SAVE is enabled
    if SAVE:
        os.makedirs(os.path.dirname(target_sequence_path), exist_ok=True)
        with open(target_sequence_path, "w") as f:
            f.write(" ".join(map(str, target_sequence.tolist())))
        print(f"Target sequence saved to {target_sequence_path}")



    # Load or generate samples for supervised fine-tuning
    if LOAD and os.path.exists(samples_path):
        # Load samples from file
        samples = []
        with open(samples_path, "r") as f:
            for line in f:
                samples.append(torch.tensor(list(map(int, line.strip().split())), dtype=torch.long, device=device))
        print("Loaded generated samples from file.")
    else:
        # Generate new samples
        samples = generate_gamma_based_samples(policy_net, target_sequence, gamma, sample_size=3200)



    # Supervised Fine-Tuning (SFT) Phase
    print("Starting Supervised Fine-Tuning (SFT)...")
    supervised_fine_tuning(samples, policy_net, iterations=sft_iterations, batch_size=batch_size, lr=learning_rate)
    acc_sft = pass_at_n_evaluation(policy_net, target_sequence, N=1000)
    print(f"after initialization: {acc_sft:.4f}")

    # Reinforcement Learning Phase
    print("Starting Reinforcement Learning...")
    train_policy(policy_net, target_sequence, iterations=rl_iterations, batch_size=batch_size, lr=learning_rate, num_rollouts=num_rollouts)

