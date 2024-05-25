import matplotlib.pyplot as plt
import os
import json

noise_level_list = [0, 5, 10, 20, 30]  

file_noise_level_str = ['0', '05', '10', '20', '30']

llama_chat_out_path = '../outputs/llama_chat_2'

vicuna_out_path = '../outputs/vicuna_2'

prompt_level_list = [1, 3, 5]

exact_match = {}
f1 = {}



#vicuna
exact_match['vicuna'] = {}
f1['vicuna'] = {}
for k in prompt_level_list:
    exact_match['vicuna'][k] = {}
    f1['vicuna'][k] = {}
    for i in range(len(noise_level_list)):
        file_name = 'vicuna_noise' + file_noise_level_str[i] + '_b16_qa_'+ str(k)+ 'prompt'
        cur_path = combined_path = os.path.join(vicuna_out_path, file_name, 'results.json')
        
        with open(cur_path, 'r') as f:
            data = json.load(f)
            exact_match['vicuna'][k][noise_level_list[i]] = data['exact_match']
            f1['vicuna'][k][noise_level_list[i]] = data['f1']

# print(exact_match)
# print(f1)

PRINT_VICUNA = True

if PRINT_VICUNA:
    for i in prompt_level_list:
        exact_match_values = list(exact_match['vicuna'][i].values())
        plt.plot(noise_level_list, exact_match_values, label=f'{i}shots', marker='o')
        plt.legend()
        plt.title(f'few-shots vicuna')
        plt.xlabel('noise_level')
        plt.ylabel('exact match score')
    plt.savefig(f'./pics/vicuna_exact_match.png')
    plt.clf()

    for i in prompt_level_list:
        f1_values = list(f1['vicuna'][i].values())
        plt.plot(noise_level_list, f1_values, label=f'{i}shots', marker='o')
        plt.legend()
        plt.title(f'few-shots vicuna')
        plt.xlabel('noise_level')
        plt.ylabel('f1 score')
    plt.savefig(f'./pics/vicuna_f1.png')
    plt.clf()