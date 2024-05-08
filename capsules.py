from typing import List
from torch import randn
import torch

def run_one_layer(num_capsules, input, layer, layer_parity):
	outputs = []
	for i in range(num_capsules):
		input_size = input.shape[0] // num_capsules
		view_start = i * input_size
		view_end = i * input_size + input_size
		input_view = input[view_start:view_end]
		# TODO: write directly into the proper spots in the output tensor rather than concatenating and rearranging
		output = layer(input_view)
		outputs.append(output)
	if layer_parity:
		outputs.append(outputs[0][:1])
		return torch.concat(outputs)[1:]
	else:
		outputs.insert(0, outputs[-1][-1:])
		return torch.concat(outputs)[:-1]

def run_all_layers_in_one_column(input, layers, num_capsules_tall):
	previous_output = input
	for i, layer in enumerate(layers):
		previous_output = run_one_layer(num_capsules_tall, previous_output, layer, i % 2)
	return previous_output

def run_all_columns(input: torch.Tensor, layers: List[torch.nn.Module], num_capsules_wide: int, feature_sizes: List[int]):
	assert feature_sizes[0] == feature_sizes[-1], f"First ({feature_sizes[0]}) and last ({feature_sizes[-1]}) feature sizes need to be the same."
	assert len(input.shape) == 1, "Input tensor expected to be a single dimension."
	assert input.shape[0] % feature_sizes[0] == 0, f"Input tensor expected to have a size ({input.shape[0]}) that divides evenly by the first feature size ({feature_sizes[0]})."

	num_capsules_tall = input.shape[0] // feature_sizes[0]
	previous_column_output = input
	for _ in range(num_capsules_wide):
		previous_column_output = run_all_layers_in_one_column(previous_column_output, layers, num_capsules_tall)
	return previous_column_output



feature_sizes = [5, 7, 13, 9, 5]
layers = []
for i in range(len(feature_sizes)-1):
	layers.append(torch.nn.Sequential(torch.nn.Linear(feature_sizes[i], feature_sizes[i+1], device = 'cuda'), torch.nn.ReLU()).forward)
input = randn(feature_sizes[0] * 3, requires_grad = False, device = 'cuda')
result = run_all_columns(input, layers, 4, feature_sizes)
