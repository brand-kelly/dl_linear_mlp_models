## MLP Train 5

## MLP Train 6 - Passed grader

self.fc_in = nn.Linear(in_features = 3*64*64, out_features = 256)
self.relu = nn.ReLU()
self.fc2 = nn.Linear(in_features = 256, out_features = 128)
self.fc_out = nn.Linear(in_features = 128, out_features = 6)

15 epochs
Cross Entropy Loss

## MLP Train 7 - passed grader

Used 2 layers,
self.fc_in = nn.Linear(in_features = 3*64*64, out_features = 64)
self.relu = nn.ReLU()
self.fc2 = nn.Linear(in_features = 64, out_features = 32)
self.fc_out = nn.Linear(in_features = 32, out_features = 6)

Adam with lr = 0.001

15 epochs
Cross Entropy Loss

## MLP Train 8 - 27/30 grader

self.fc_in = nn.Linear(in_features = 3*64*64, out_features = 50)
self.relu = nn.ReLU()
self.fc_out = nn.Linear(in_features = 50, out_features = 6)

SGD:
lr = 0.01
momentum = 0.9
weight_decay = 0.1e-4

15 epochs
Cross Entropy Loss

## MLP Train 9 - 28/30 grader

self.fc_in = nn.Linear(in_features = 3*64*64, out_features = 50)
self.relu = nn.ReLU()
self.fc_out = nn.Linear(in_features = 50, out_features = 6)

SGD:
lr = 0.01
momentum = 0.9
weight_decay = 0.1e-4

20 epochs
Cross Entropy Loss

## MLP Train 10 - passed

self.fc_in = nn.Linear(in_features = 3*64*64, out_features = 50)
self.relu = nn.ReLU()
self.fc_out = nn.Linear(in_features = 50, out_features = 6)

SGD:
lr = 0.01
momentum = 0.9
weight_decay = 0.1e-4

20 epochs
Cross Entropy Loss

Moved the loading of the training data inside of the epoch for loop to shuffle every time.
