import torch
from fancy_einsum import einsum
from tqdm import tqdm
import numpy as np
import heapq

def head_attribution_over_all_data(model, data, device, neurons, batch_size=8):

    head_attribution_dict = {}

    neuron_prompt_head_scores = {}
    for neuron in tqdm(neurons):
        # Load and Truncate Prompts
        trunc_prompts, prompts_metadata = data.load_truncated_prompts(model, neuron)
        trunc_prompts = trunc_prompts[:20] # 20
        # trunc_prompts = trunc_prompts[:10] # 10
        
        print(len(trunc_prompts))

        num_batches = len(trunc_prompts) // batch_size + (1 if len(trunc_prompts) % batch_size != 0 else 0)

        head_results = {}
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(trunc_prompts))
            current_batch = trunc_prompts[start_idx:end_idx]

            # Run head attribution
            tokens = model.to_tokens(current_batch, prepend_bos=True).to(device=device)
            original_logits, cache = model.run_with_cache(tokens, )

            # Prepare prompts by heads
            head_attribution = _get_head_attribution(model, cache, tokens, neuron) # Gives [batch, head]
            # _, top_heads = torch.topk(head_attribution, k=3, dim=-1)

            if str(neuron) not in neuron_prompt_head_scores:
                neuron_prompt_head_scores[str(neuron)] = {}

            for i, prompt in enumerate(current_batch):

                if prompt not in neuron_prompt_head_scores[str(neuron)]:
                    neuron_prompt_head_scores[str(neuron)][prompt] = {}

                for all_heads_idx, score in enumerate(head_attribution[i].cpu().tolist()):
                    neuron_prompt_head_scores[str(neuron)][prompt][all_heads_idx] = score
    
            sigma = head_attribution.std(dim=-1)
            mean = head_attribution.mean(dim=-1)
            # top_heads_2 = head_attribution[sigma > 2]
            top_heads_2 = torch.where(head_attribution > (mean + 2*sigma).unsqueeze(-1), torch.ones_like(head_attribution), torch.zeros_like(head_attribution))
            
            top_heads_2_index = torch.nonzero(top_heads_2, as_tuple=True)
            indices, values = top_heads_2_index
            top_heads_2_list = [[] for _ in range(len(current_batch))] 
            if len(indices) == 0:
                top_heads_2_list.append([])
            else:
                for index in indices.unique():
                    top_heads_2_list.append(values[indices == index].tolist())

            for i, prompt in enumerate(current_batch):
                head_results[prompt] = top_heads_2_list[i]

        head_attribution_dict[str(neuron)] = head_results

    return head_attribution_dict, neuron_prompt_head_scores


def _get_head_attribution(model, cache, tokens, neuron):
    # Get prompt lengths
    if "pythia" in model.cfg.model_name:
        pad_token = 1
        bos_token = 0
    else:
        pad_token = bos_token = 50256 # This is true for GPT-2
    prompt_lengths = (torch.logical_and(tokens != pad_token, tokens != bos_token)).sum(dim=-1)

    # Get the correct last-seq for each prompt (since they are padded, the last seq position differs for each prompt)
    head_output = cache.stack_head_results()
    expanded_index_tensor = prompt_lengths.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(head_output.shape[0], head_output.shape[1], 1, head_output.shape[-1])
    head_output_last_seq = torch.gather(head_output, 2, expanded_index_tensor).squeeze(2)

    # Get dot product of neuron with each head's output
    layer, no = neuron
    neuron_w_in = model.W_in[layer, :, no]
    head_attribution = einsum("head batch weight, weight->batch head", head_output_last_seq, neuron_w_in)
    head_attribution = head_attribution[:, :(layer+1)*model.cfg.n_heads] # Filter for only heads before the neuron
    return head_attribution # Shape [batch, head]


# Activation Steering Based Head Attribution

def activation_steering_based_head_attribution(model, data, device, neurons, total_active_heads, boost_factor=2, batch_size=8):
    model.cfg.use_attn_result=True # This will allow me to boost the activation heads
   
    max_pq_activation = []
    neuron_prompt_head_scores = {}
    last_seq_neuron_activations = {}
    last_seq_neuron_activations_boosted_head = {} 
    head_attribution_dict = {} 
    #cnt = 0
    for neuron in tqdm(neurons):
        torch.cuda.empty_cache()
        #cnt += 1
        neuron_layer, neuron_idx = neuron

        # Load and Truncate Prompts
        trunc_prompts, prompts_metadata = data.load_truncated_prompts(model, neuron)
        trunc_prompts = trunc_prompts[:20] # 20
        #print(len(trunc_prompts))


        num_batches = len(trunc_prompts) // batch_size + (1 if len(trunc_prompts) % batch_size != 0 else 0)

        batched_tokens = [torch.tensor([]) for _ in range(num_batches)]

        """
        1. Compute activations prior to head intervention
            last_seq_neuron_activations format:
            last_seq_neuron_activations[neuron i]: [(prompt_idx, activation)]
        """
        if neuron not in last_seq_neuron_activations:
            last_seq_neuron_activations[neuron] = {}
        def get_neurons_activations_hook(neuron,start_idx, end_idx, prompts_lengths):
            neuron_layer, neuron_idx = neuron
            def hook_fn(act, hook):
                batch_size = end_idx - start_idx

                # act shape is (batch_size, seq_len, d_model)
                neuron_activations = [(i, act[i % batch_size, prompts_lengths[i%batch_size], neuron_idx].item()) for i in range(start_idx, end_idx, 1)]
               
                for (prompt_idx, prompt_act) in neuron_activations:
                    last_seq_neuron_activations[neuron][prompt_idx] = prompt_act

                return act

            return hook_fn
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(trunc_prompts))
            current_batch = trunc_prompts[start_idx:end_idx]

            # Run head attribution
            tokens = model.to_tokens(current_batch, prepend_bos=True).to(device=device)
            batched_tokens[batch_idx] = tokens
            pad_token = bos_token = 50256 # This is true for GPT-2

            prompts_lengths = [(tokens[i]!=torch.tensor(tokens[i].shape[0] * [pad_token]).to(device)).sum().item() for i in range(tokens.shape[0])]

            # Save neuron activations
            hooks = [(f"blocks.{neuron_layer}.mlp.hook_post", get_neurons_activations_hook(neuron, start_idx, end_idx, prompts_lengths))]
            original_logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        
        if str(neuron) not in head_attribution_dict:
            head_attribution_dict[str(neuron)] = {}
        """
        1. Compute activations after interventions in valid heads for each neuron
            last_seq_neuron_activations_boosted_head format:
            last_seq_neuron_activations_boosted_head[neuron i][head j]: {(prompt_idx, activation) 
        """
        def boost_head_hook(head, boost_factor, prompts_lengths):
            def hook_fn(act, hook):
                for b, prompt_length in enumerate(prompts_lengths):
                    # act shape is (batch_size, seq_len, heads, d_model)
                    act[b, prompt_length, head, :] *= boost_factor 
                return act
            return hook_fn
        
        def get_neurons_activations_boosted_head_hook(neuron, head_idx, head_layer, start_idx, end_idx, prompts_lengths):
            neuron_layer, neuron_idx = neuron
            def hook_fn(act, hook):
                batch_size = end_idx - start_idx
                # act shape is (batch_size, seq_len, d_model)
                neuron_activations = [(i, act[i % batch_size, prompts_lengths[i%batch_size], neuron_idx].item()) for i in range(start_idx, end_idx, 1)]

                for (prompt_idx, prompt_act) in neuron_activations:
                    last_seq_neuron_activations_boosted_head[neuron][(head_idx, head_layer)][prompt_idx] = prompt_act
                return act

            return hook_fn

        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads
        neuron_layer, neuron_idx = neuron
        for head_layer in range(neuron_layer+1): # paper says <= neuron_layer, base code says < neuron_layer
            for head_idx in range(n_heads):
                # Initialising used dictionaries
                if neuron not in last_seq_neuron_activations_boosted_head:
                    last_seq_neuron_activations_boosted_head[neuron] = {}
                if (head_idx, head_layer) not in last_seq_neuron_activations_boosted_head[neuron]:
                    last_seq_neuron_activations_boosted_head[neuron][(head_idx, head_layer)] = {}
                # Run the model again, boost the head's activation and save the neuron's activation
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(trunc_prompts))

                    # Run head attribution
                    tokens = batched_tokens[batch_idx]
                    pad_token = bos_token = 50256 # This is true for GPT-2
                    prompts_lengths = [(tokens[i]!=torch.tensor(tokens[i].shape[0] * [pad_token]).to(device)).sum().item() for i in range(tokens.shape[0])]

                    hooks = [
                        (f"blocks.{head_layer}.attn.hook_result", boost_head_hook(head_idx, boost_factor, prompts_lengths)), 
                        (f"blocks.{neuron_layer}.mlp.hook_post", get_neurons_activations_boosted_head_hook(neuron, head_idx, head_layer, start_idx, end_idx, prompts_lengths))
                    ]

                    logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
                
        # Compute the active and inactive heads for a given prompt and a given neuron
        """
        1. Compute the delta of heads
            head_attribution_dict:
            head_attribution_dict[neuron i][prompt j] 
        """

        if str(neuron) not in neuron_prompt_head_scores:
            neuron_prompt_head_scores[str(neuron)] = {}

        for (prompt_idx, prompt) in enumerate(trunc_prompts):
            current_prompt_head_acts = []

            if prompt not in neuron_prompt_head_scores[str(neuron)]:
                neuron_prompt_head_scores[str(neuron)][prompt] = {}

            if prompt not in head_attribution_dict[str(neuron)]:
                head_attribution_dict[str(neuron)][prompt] = []

            for (head_idx, head_layer) in last_seq_neuron_activations_boosted_head[neuron]:
                # Linearizing the count of heads
                all_heads_idx = head_layer*n_heads + head_idx
                baseline_act = last_seq_neuron_activations[neuron][prompt_idx]
                new_act = last_seq_neuron_activations_boosted_head[neuron][(head_idx, head_layer)][prompt_idx]
                delta_act = new_act - baseline_act
                current_prompt_head_acts.append((all_heads_idx, delta_act))
                neuron_prompt_head_scores[str(neuron)][prompt][all_heads_idx] = delta_act

            activations = [act for (_, act) in current_prompt_head_acts]
            mean = np.mean(activations)
            std = np.std(activations)
            print(f"--> statistics of {str(neuron)} delta activations for prompt {prompt_idx}", end=": ")
            print("mean", end=": ")
            print(mean, end="/ ")
            print("std", end=": ")
            print(std, end="/ ")
            print("positive_deltas", end=": ")
            print(len([dontcare for dontcare in activations if dontcare > 0]), end="/ ")
            print("negative/null_deltas", end=": ")
            print(len([dontcare for dontcare in activations if dontcare <= 0]))
            print(f"activations of neuron {str(neuron)} for prompt_idx {prompt_idx}", end=": ")
            print(sorted(activations, reverse=True))

            for (all_heads_idx, prompt_act) in current_prompt_head_acts:
                heapq.heappush(max_pq_activation, (-prompt_act, str(neuron), prompt, all_heads_idx))  # priority queue containing all activation deltas ratio

    """
    1. Select the set of active heads
    """
    while(max_pq_activation and total_active_heads>0):
        delta_act, neuron, prompt, all_heads_idx = heapq.heappop(max_pq_activation)
        if(delta_act>=0):
            break
        head_attribution_dict[str(neuron)][prompt].append(all_heads_idx)
        total_active_heads -= 1
        
    return head_attribution_dict, neuron_prompt_head_scores, last_seq_neuron_activations, last_seq_neuron_activations_boosted_head, total_active_heads