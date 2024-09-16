import matplotlib.pyplot as plt
import os
import json

noise_level_list = [0, 5, 10, 20, 30, 50, 75, 100]  

file_noise_level_str = ['0', '05', '10', '20', '30', '50', '75', '100']

llama_chat_out_path = '../outputs/llama_chat_2'

vicuna_out_path = '../outputs/vicuna_2'

prompt_level_list = [1, 3, 5]

exact_match = {}
f1 = {}


def draw_pics(model_name_str, model_out_path):
    exact_match[model_name_str] = {}
    f1[model_name_str] = {}
    for k in prompt_level_list:
        exact_match[model_name_str][k] = {}
        f1[model_name_str][k] = {}
        for i in range(len(noise_level_list)):
            file_name = model_name_str + '_noise' +  file_noise_level_str[i] + '_b16_qa_'+ str(k)+ 'prompt'
            cur_path =  os.path.join(model_out_path, file_name, 'results.json')
            
            with open(cur_path, 'r') as f:
                data = json.load(f)
                exact_match[model_name_str][k][noise_level_list[i]] = data['exact_match']
                f1[model_name_str][k][noise_level_list[i]] = data['f1']

    for i in prompt_level_list:
        exact_match_values = list(exact_match[model_name_str][i].values())
        plt.plot(noise_level_list, exact_match_values, label=f'{i} shots', marker='o')
        plt.legend()
        plt.title(f'few-shots {model_name_str}')
        plt.xlabel('noise_level')
        plt.ylabel('exact match score')
    plt.savefig(f'./pics/{model_name_str}_exact_match.png')
    plt.clf()

    for i in prompt_level_list:
        # f1_values = list(map(lambda x: x*100,  f1[model_name_str][i].values()))
        f1_values = list( f1[model_name_str][i].values())
        plt.plot(noise_level_list, f1_values, label=f'{i} shots', marker='o')
        plt.legend()
        plt.title(f'few-shots {model_name_str}')
        plt.xlabel('noise_level')
        plt.ylabel('f1 score')
    plt.savefig(f'./pics/{model_name_str}_f1.png')
    plt.clf()



model_name_list = ['vicuna', 'llama_chat']
model_out_path_list = [vicuna_out_path, llama_chat_out_path]


for i,name in enumerate(model_name_list):
    draw_pics(name, model_out_path_list[i])

print(exact_match)
print(f1)