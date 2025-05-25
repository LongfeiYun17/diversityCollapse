import json
# pass 1, ... pass 32 
data_file = 'alpaca_eval/data/alpaca_eval_outputs_llama_full_template_subset_minimum_dialog.json'

with open(data_file, 'r') as f:
    data = json.load(f)

for pass_num in [1, 4, 8, 16, 32]:
    outputs = []
    for item in data:
        # select the response based on the scores
        scores = item['scores'][:pass_num]
        responses = item['responses'][:pass_num]
        # select the response with the highest score
        response = responses[scores.index(max(scores))]
        outputs.append({
            'instruction': item['prompt'],
            'output': response,
            'generator': f'llama_full_template_subset_minimum_dialog_pass_{pass_num}_v0.1',
        })
    # save the outputs
    with open(f'alpaca_eval/data/alpaca_eval_outputs_llama_full_template_subset_minimum_dialog_pass_{pass_num}.json', 'w') as f:
        json.dump(outputs, f)


