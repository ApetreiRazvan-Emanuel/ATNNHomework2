# ATNNHomework2
Apetrei Razvan-Emanuel, IAO1

Repository pentru partea prima parte a temei cu training experiments.

| Experiments                                               | Train Transformations                      | Test Acc. |
|-----------------------------------------------------------|--------------------------------------------|-----------|
| 1. SGD, lr 0.1, momentum 0.9, CosineAnnealingLR           | Crop, Horizontal                           | 74.33%    |
| 2. SGD, lr 0.01, momentum 0.85, ReduceLROnPlateau         | Crop, Horizontal, Random Erasing           | 71.45%    |
| 3. SGD, lr 0.1, momentum 0.9, Nesterov, ReduceLROnPlateau | Crop, Horizontal, Vertical, Random Erasing | 70.84%    |
| 4. Adam, lr 0.001, CosineAnnealingLR                      | Crop, Horizontal, Random Erasing           | 69.78%    |

The files are saved on this repository, more precisely each has a folder: Experiment1, Experiment2, Experiment3, Experiment4 
containing the source code and submission.csv