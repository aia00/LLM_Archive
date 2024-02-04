import pandas as pd 
df = pd.read_csv('Llama_2/experiments/ICL_DEMO/ICL_demo_data.csv')
# print(df)

def concat_questions(data):
    result = ""
    for i, (_, row) in enumerate(data.iterrows()):
        question = row['Question']
        answer = row['Answer']
        if i != 2: # for the first and second question, add both question and answer
            result += f"Question: {question}; Answer: {answer}. "
        else: # for the third question, add only the question
            result += f"Question: {question}; Answer:"
    return result

# Function to split data into chunks of three
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

# Apply the function on chunks of each group of type
output_data = []
for name, group in df.groupby('Type'):
    for chunk in chunker(group, 3):
        output_data.append([name, concat_questions(chunk)])

new_df = pd.DataFrame(output_data, columns=['Type', 'Sentence'])

# Save the new dataframe to a new csv file
new_df.to_csv('Llama_2/experiments/ICL_DEMO/processed_data.csv', index=False)