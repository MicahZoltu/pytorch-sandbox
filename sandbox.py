import torch
from torch.nn.functional import cross_entropy

output = torch.transpose(torch.tensor([
	[
		[0.9998, 0.0001, 0.0001],
		[0.0001, 0.9998, 0.0001],
		[0.0001, 0.0001, 0.9998],
		[0.9998, 0.0001, 0.0001],
		[0.0001, 0.9998, 0.0001],
	],
	[
		[0.0001, 0.9998, 0.0001],
		[0.0001, 0.9998, 0.0001],
		[0.9998, 0.0001, 0.0001],
		[0.0001, 0.0001, 0.9998],
		[0.0001, 0.0001, 0.9998],
	],
	[
		[0.9998, 0.0001, 0.0001],
		[0.9998, 0.0001, 0.0001],
		[0.9998, 0.0001, 0.0001],
		[0.9998, 0.0001, 0.0001],
		[0.9998, 0.0001, 0.0001],
	],
	[
		[0.9998, 0.0001, 0.0001],
		[0.0001, 0.0001, 0.9998],
		[0.9998, 0.0001, 0.0001],
		[0.0001, 0.0001, 0.9998],
		[0.9998, 0.0001, 0.0001],
	],
], requires_grad=True), 1, 2).log()

exact = torch.tensor([
	[ 0, 1, 2, 0, 1 ],
	[ 1, 1, 0, 2, 2 ],
	[ 0, 0, 0, 0, 0 ],
	[ 0, 2, 0, 2, 0 ],
], dtype=torch.int64)
one_wrong_letter = torch.tensor([
	[ 0, 1, 2, 0, 1 ],
	[ 1, 1, 0, 2, 2 ],
	[ 0, 1, 0, 0, 0 ],
	[ 0, 2, 0, 2, 0 ],
], dtype=torch.int64)
one_wrong_batch = torch.tensor([
	[ 0, 1, 2, 0, 1 ],
	[ 1, 1, 0, 2, 2 ],
	[ 1, 1, 1, 1, 1 ],
	[ 0, 2, 0, 2, 0 ],
], dtype=torch.int64)
one_wrong_letter_per_batch = torch.tensor([
	[ 1, 1, 2, 0, 1 ],
	[ 2, 1, 0, 2, 2 ],
	[ 1, 0, 0, 0, 0 ],
	[ 0, 0, 0, 2, 0 ],
], dtype=torch.int64)
all_wrong = torch.tensor([
	[ 1, 2, 0, 1, 2 ],
	[ 2, 2, 1, 0, 0 ],
	[ 1, 1, 1, 1, 1 ],
	[ 1, 0, 1, 0, 1 ],
], dtype=torch.int64)

red = '\033[31m'
green = '\033[32m'
reset = '\033[0m'
print(f'Output Shape: {red}{output.shape}{reset}')
print(f'Target Shape: {red}{exact.shape}{reset}')
print(f'Exact Match: {green}{cross_entropy(output, exact, reduction="mean")}{reset}')
print(f'One Wrong Letter: {green}{cross_entropy(output, one_wrong_letter, reduction="mean")}{reset}')
print(f'One Wrong Batch: {green}{cross_entropy(output, one_wrong_batch, reduction="mean")}{reset}')
print(f'One Wrong Letter Per Batch: {green}{cross_entropy(output, one_wrong_letter_per_batch, reduction="mean")}{reset}')
print(f'All Wrong: {green}{cross_entropy(output, all_wrong, reduction="mean")}{reset}')
