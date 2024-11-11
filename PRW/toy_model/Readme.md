# Toy Problem Setup
## 1. Task

The vocab dictionary is integers ranging from [1, 15]

We are given a specific target sequence of length 5 ranging from [1, 14], such as [12, 3, 5, 8, 4], call it tar_seq.

Now our job is to learn a policy that can generate a sequence that exactly contains tar_seq. This generated sequence should be terminated whenever seeing a 15 or reaching a max_length of 10.

## 2. Policy Net

This is a 3-layer fully connect neuron network.

## 3. Initialization:



## 4. Policy update

Follow this formula
![alt text](pics/PAV_FORMULA.png)

Now for the Q function part, I use the monte-carlo method to have 100 rollouts following current policy, and calculate a mean as the Q value of the current step.

sdef ** fre ** 