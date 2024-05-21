import os
import torch
import numpy as np
from tqdm.auto import tqdm
from omegaconf import DictConfig
import hydra
import json
from eval_scripts.eval_tqa import get_eval_score
from datasets import load_dataset
from utils.common_utils import parse_dtype, chunks
from utils.normalization_utils import trim_prediction
from utils.generation_utils import stop_sequences_criteria
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
    
access_token = "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"

@hydra.main(config_path='configs', config_name='llama_tqa_0.yaml', version_base='1.3')
def main(cfg:DictConfig):
    set_seed(cfg.run.seed)
    outputs = []
    exact_match_list = []
    f1_list = []
    
    ## TODO: LOAD MODEL AND TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_path, token=access_token)
    tokenizer.padding_side = 'left'
    stop = [tokenizer.eos_token]
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_path,
        device_map="auto",
        torch_dtype=parse_dtype(cfg.model.torch_dtype),
        token=access_token
    ).eval()
    
    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:   # llama-2
        tokenizer.pad_token_id = tokenizer.unk_token_id
        model.config.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id
    # print(f'model.config.pad_token_id: {model.config.pad_token_id}')
    # print(f'tokenizer.pad_token_id: {tokenizer.pad_token_id}')
    
    ## TODO: LOAD DATASET
    if cfg.dataset.subset:
        data = load_dataset(cfg.dataset.name, cfg.dataset.subset)
    else:
        data = load_dataset(cfg.dataset.name)
    data = data[cfg.dataset.split]
    
    if 'trivia_qa' in cfg.dataset.name:
        eval_questions = [e['question'] for e in data]
        eval_answers = [e['answer'] for e in data]
        stop += ["\n", ",", "."]                                      # stopping criteria
        
    
    questions_chunks = list(chunks(eval_questions, cfg.run.batch_size))
    answers_chunks = list(chunks(eval_answers, cfg.run.batch_size))
    
    
    '''        
    Design options:
        1. keep track of max_length and start, end indices -> deduct post_padding index
        2. only record start, end indices -> use find() to get the first non-padding index and deduct post_padding index (adopted)
    '''
    # conv_template = get_conversation_template(cfg.model.conversation_template)
    
    pbar = tqdm(range(len(questions_chunks)))
    for chunk_id in pbar:
        questions_chunk = questions_chunks[chunk_id]
        answers_chunk = answers_chunks[chunk_id]
        
        full_prompts = []
        start_token_indices = []
        end_token_indices = []
        
        for i, question in enumerate(questions_chunk):
            question = question.replace('?', '')
            llama_chat_prompt = '''
                Example Question: What is the capital city of France?
                Example Answer: Paris.
                Example Question: Who wrote the play 'Romeo and Juliet'?
                Example Answer: William Shakespeare.
                Example Question: What is the largest ocean on Earth?
                Example Answer: The Pacific Ocean.
                Example Question: In what year did humans first land on the Moon?
                Example Answer: 1969.
                Example Question: What is the chemical symbol for gold?
                Example Answer: Au.
                '''
            full_prompt = f"Answer these questions:\nQ: {question}?\nA:"
            full_prompts.append(full_prompt)
            
            start_string_idx = full_prompt.find(question)
            end_string_idx = full_prompt.find('?')                                      # make sure there is no question mark in the system prompt
            
            # print(f'pre-question text: {full_prompt[:start_string_idx]}')
            # print(f'encoded: {tokenizer.encode(full_prompt[:start_string_idx])}')
            start_token_idx = len(tokenizer.encode(full_prompt[:start_string_idx]))
            end_token_idx = len(tokenizer.encode(full_prompt[:end_string_idx])) + 1       # index of '?', add 1 to include noise on '?'
            start_token_indices.append(start_token_idx)
            end_token_indices.append(end_token_idx)

        inputs = tokenizer.batch_encode_plus(
            full_prompts,
            padding='longest',
            truncation=True,
            max_length=cfg.run.max_length,
            return_tensors='pt',
            return_attention_mask=True,
        )
        
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            tokenizer, stop, inputs.input_ids.shape[1], inputs.input_ids.shape[0]
        )

        # edit noise instance
        noise_scale = torch.zeros_like(inputs.input_ids, dtype=torch.float32)
        for i in range(inputs.input_ids.shape[0]):
            attention_mask = inputs.attention_mask[i]
            start_idx = start_token_indices[i]
            end_idx = end_token_indices[i]
            
            # offset = index of the bos token + 1 (e.g. no padding => offset = 1)
            #          index of the bos token = index of the first nonzero element in the attention mask
            offset = torch.nonzero(attention_mask)[0] + 1
            offset = offset.item()
            noise_scale[i, (offset+start_idx):(offset+end_idx)] = cfg.run.noise_scale
        
            # same adjusted end index over all sequences in batch
            # print(f'*** adjusted end index: {offset+end_idx} ***')
        
        # print(f'input_ids in main_qa.py: {inputs.input_ids}')
        
        # call generate with noise (prompt, noise)
        with torch.inference_mode():
            predictions = model.generate(
                inputs=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=cfg.run.max_new_tokens,
                noise_scale=noise_scale,
                # stopping_criteria=stopping_criteria,
                do_sample=False,
                temperature=0.0,
                use_cache=True,
            )
            
        predictions = tokenizer.batch_decode(
            predictions[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        predictions = trim_prediction(predictions)
        # print(f'predictions: {predictions}')
        
        for i, (prediction, answer) in enumerate(zip(predictions, answers_chunk)):
            eval_score_dict = get_eval_score(prediction, answer)
            # print('------------------------------')
            # print(f'prediction: {prediction}')
            # print(f'answer: {answer}')
            # print(f"Eval scores: {eval_score_dict}")
            
            outputs.append({
                'question': questions_chunk[i],
                'prediction': prediction,
                'answer_aliases': answer.get('normalized_aliases', []),
                'eval_score': eval_score_dict
            })
            exact_match_list.append(eval_score_dict['exact_match'])
            f1_list.append(eval_score_dict['f1'])
        
        pbar.set_description(
            f'em = {np.mean(exact_match_list):.3f}  f1 = {np.mean(f1_list):.3f}'
        )
        
    results = {
        'dataset': cfg.dataset.name,
        'subset': cfg.dataset.subset if cfg.dataset.subset else 'N/A',
        'split': cfg.dataset.split,
        'exact_match': np.mean(exact_match_list),
        'f1': np.mean(f1_list)
    }
    
    print(results)
    
    with open(os.path.join( hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f)
        
    with open(os.path.join( hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'outputs.json'), 'w') as f:
        json.dump(outputs, f)
    

if __name__ == '__main__':
    main()
    
    
    
'''
ex)
python main_hydra.py
    result is saved in outputs
python main_hydra.py -m model=llama2,vicuna smoothllm_perturbation_rate=1,5 smoothllm_num_copies=3
    results are saved in multirun
    
'''