import math
import pickle
import statistics
import torch
from datetime import timedelta
from time import perf_counter
from typing import Callable
from functools import partial, reduce
from torch.nn.functional import conv2d, relu, adaptive_avg_pool2d, cross_entropy, softmax
from torch.utils.data import DataLoader, TensorDataset

RED = '\033[31m'
GREEN = '\033[32m'
RESET = '\033[0m'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
random_tensor = partial(torch.randn, device=device)
empty_tensor = partial(torch.empty, device=device)
zero_tensor = partial(torch.zeros, device=device)
tensor = partial(torch.tensor, device=device)

def get_model_and_parameters(possible_predictions: int, width: int, height: int):
	parameters = []
	def generate_2d_convolutional_layer(num_input_channels: int, num_output_channels: int, kernel_size: int,num_groups: int,stride: int, padding: int):
		nonlocal parameters
		weight_tensor = empty_tensor(num_output_channels, num_input_channels // num_groups, kernel_size, kernel_size)
		bias_tensor = empty_tensor(num_output_channels)
		bound = math.sqrt(num_groups / (num_input_channels * kernel_size * kernel_size))
		weight_parameter = torch.nn.Parameter(torch.nn.init.uniform_(weight_tensor, -bound, bound))
		bias_parameter = torch.nn.Parameter(torch.nn.init.uniform_(bias_tensor, -bound, bound))
		parameters.append(weight_parameter)
		parameters.append(bias_parameter)
		return lambda x: relu(conv2d(x, weight_parameter, bias_parameter, stride, padding))
	layers = [
		lambda x: x.view(-1, 1, width, height),
		generate_2d_convolutional_layer(num_input_channels=1, num_output_channels=16, kernel_size=3, num_groups=1, stride=2, padding=1),
		generate_2d_convolutional_layer(num_input_channels=16, num_output_channels=16, kernel_size=3, num_groups=1, stride=2, padding=1),
		generate_2d_convolutional_layer(num_input_channels=16, num_output_channels=possible_predictions, kernel_size=3, num_groups=1, stride=2, padding=1),
		lambda x: adaptive_avg_pool2d(x, 1),
		lambda x: x.view(-1, possible_predictions),
	]
	def execute_layer(previous_layer_output: torch.Tensor, layer: Callable[[torch.Tensor], torch.Tensor]):
		return layer(previous_layer_output)

	def model(input_batch: torch.Tensor):
		return reduce(execute_layer, layers, input_batch)

	return model, parameters

def calculate_loss(output_batch: torch.Tensor, expected_batch: torch.Tensor):
	return cross_entropy(output_batch, expected_batch)

def check_batch_accuracy(output_batch: torch.Tensor, expected_batch: torch.Tensor):
	# get the probability the model estimated for the expected output for each item in the batch
	probability_of_expecteds = [softmax(output_batch[i], dim=-1)[expected.item()] for i, expected in enumerate(expected_batch)]
	# return the average of all of the probabilities
	return statistics.fmean(probability_of_expecteds)

def train_on_one_batch(runner: Callable[[torch.Tensor], torch.Tensor], optimizer: torch.optim.SGD, input_batch: torch.Tensor, expected_batch: torch.Tensor):
	output_batch = runner(input_batch)
	loss = calculate_loss(output_batch, expected_batch)
	# calculate the gradient updates (weights and biases)
	loss.backward()
	# apply the gradient updates based on the learning rate
	optimizer.step()
	optimizer.zero_grad()

	return output_batch, loss

def train(runner: Callable[[torch.Tensor], torch.Tensor], optimizer: torch.optim.SGD, dataloader: DataLoader, epochs: int = 1):
	previous_average_loss = 0
	for epoch in range(epochs):
		epoch_start = perf_counter()
		batch_seconds = []
		losses = []
		for input_batch, expected_batch in dataloader:
			batch_start = perf_counter()
			output_batch, loss = train_on_one_batch(runner, optimizer, input_batch, expected_batch)
			batch_seconds.append(perf_counter() - batch_start)
			losses.append(loss.item())
		epoch_seconds = perf_counter() - epoch_start
		average_batch_seconds = statistics.fmean(batch_seconds)
		average_loss = statistics.fmean(losses)
		print_progress(epoch, output_batch, expected_batch, average_loss, previous_average_loss, epoch_seconds, average_batch_seconds, True)
		previous_average_loss = average_loss
	print()

def validate(runner: Callable[[torch.Tensor], torch.Tensor], optimizer: torch.optim.SGD, dataloader: DataLoader):
	validation_start = perf_counter()
	batch_seconds = []
	for input_batch, expected_batch in dataloader:
		batch_start = perf_counter()
		output_batch = runner(input_batch)
		batch_seconds.append(perf_counter() - batch_start)
		print_validation_results(output_batch, expected_batch)
	validation_seconds = perf_counter() - validation_start
	average_batch_seconds = statistics.fmean(batch_seconds)
	print(f'Validation Duration: {GREEN}{timedelta(seconds=validation_seconds)}{RESET}  Average Batch Duration: {GREEN}{timedelta(seconds=average_batch_seconds)}{RESET}')

def print_progress(epoch: int, output_batch: torch.Tensor, expected_batch: torch.Tensor, average_loss: float, previous_average_loss: float, epoch_seconds: float, average_batch_seconds: float, is_epoch: bool):
	accuracy = check_batch_accuracy(output_batch, expected_batch)
	epoch_duration = timedelta(seconds=epoch_seconds)
	batch_duration = timedelta(seconds=average_batch_seconds)
	print(f'Epoch: {GREEN}{epoch}{RESET}  Loss: {GREEN if average_loss < previous_average_loss else RED}{average_loss:8.5f}{RESET}  Accuracy: {GREEN if accuracy > 0.9 else RED}{accuracy:7.5f}{RESET}  Epoch Duration: {GREEN}{epoch_duration}{RESET}  Average Batch Duration: {GREEN}{batch_duration}{RESET}', end = '\n' if is_epoch else '\r')

def print_validation_results(output_batch: torch.Tensor, expected_batch: torch.Tensor):
	for i, output in enumerate(output_batch):
		expected = expected_batch[i].item()
		probabilities = softmax(output, dim=-1)
		expected_probability = probabilities[expected].item()
		top = output.argmax().item()
		top_probability = probabilities[top].item()
		print(f'Expected: {GREEN}{expected}{RESET}  Expected Probability: {GREEN if expected_probability > 0.5 else RED}{expected_probability:7.5f}{RESET}  Top: {GREEN if top == expected else RED}{top}{RESET}  Top Probability: {GREEN if top_probability > 0.5 else RED}{top_probability:7.5f}{RESET} ')

def main():
	EPOCHS = 100
	BATCH_SIZE = 512
	LEARNING_RATE = 0.5
	MOMENTUM = 0.9
	WIDTH = 28
	HEIGHT = 28

	# load training data into memory
	with (open('./training-data/mnist.pkl', 'rb')) as file:
		((training_input, training_expected), (validation_input, validation_expected), _) = pickle.load(file, encoding="latin-1")
	# convert numpy arrays to torch tensors
	training_input, training_expected, validation_input, validation_expected = map(tensor, (training_input, training_expected, validation_input, validation_expected))
	# validate that the shape of what we loaded is sane/reasonable and matches expected height/width
	assert training_input.shape[0] == training_expected.shape[0]
	assert validation_input.shape[0] == validation_expected.shape[0]
	assert training_input.shape[1] == validation_input.shape[1]
	assert WIDTH * HEIGHT == training_input.shape[1]

	training_dataset = TensorDataset(training_input, training_expected)
	training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
	validation_dataset = TensorDataset(validation_input[0:100], validation_expected[0:100])
	validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

	possible_predictions = torch.unique(training_expected).numel()
	model, parameters = get_model_and_parameters(possible_predictions, WIDTH, HEIGHT)
	optimizer = torch.optim.SGD(parameters, LEARNING_RATE, MOMENTUM)

	train(model, optimizer, training_dataloader, EPOCHS)
	validate(model, optimizer, validation_dataloader)

start = perf_counter()
main()
print(timedelta(seconds = perf_counter() - start))
