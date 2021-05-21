Authors:

1. Haswants Aekula
2. Shreeyash Pawar
3. Smita 
4. Manjeera

We have designed a neural architecture such that:

We have designed a neural architecture such that:**Input** : an MNIST image and Random number 

**Outputs:** MNIST Digit and sum of mnist digit + random number.

![mnist-statement](https://github.com/hassiahk/EVA6-Phase1-Assignments/blob/main/Session3-Pytorch/images/mnist_statement.png)

To execute, run the _____.ipynb file or clolab link : here

**Data Representation:**

We representated dataset  as

| Image(28x28) | Label | Random_Number | Sum |
|----|----|----|----|
|[28x28]| 5 | 3 | 8 |

**Data generation:**

We generated out dataset use Dataclass from torch.utils.data as: 

```python
class MNISTRandDataset(Dataset):
    def __init__(self, mnist):
        self.mnist = mnist

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]

        rand_num = random.randint(0, 9)

        rand_num_tensor = F.one_hot(torch.tensor(rand_num), num_classes=10)
        sum_label = label + rand_num

        return image, rand_num_tensor, label, sum_label
```

**Data Input Combination:**

We have concatenated the final 10 bit embeddings of MNIST image from FC layer with the Random number represented in 10-bit 1-hot encoding.

We get 20 size tensor, which we pass through final layer of 20x19, to get all possible 19 sums(0 → 18)

![input combination](https://github.com/hassiahk/EVA6-Phase1-Assignments/blob/main/Session3-Pytorch/images/Selection_188.png)

Here: out → MNIST digit, sum_out → MNIST digit + out

Model architecture:

![model arch](https://github.com/hassiahk/EVA6-Phase1-Assignments/blob/main/Session3-Pytorch/images/Selection_189.png)

**Loss Function:**

We used 2 losses , one for each ouput , and averaged them to get total loss.

The individual losses used were "cross_entropy" loss.

Since this is classification task, where we had to classify numbers for both the outputs, and hence use entropy loss as:

![CE1](https://github.com/hassiahk/EVA6-Phase1-Assignments/blob/main/Session3-Pytorch/images/CE1.png)

![CE2](https://github.com/hassiahk/EVA6-Phase1-Assignments/blob/main/Session3-Pytorch/images/CE2.png)

![CE3](https://github.com/hassiahk/EVA6-Phase1-Assignments/blob/main/Session3-Pytorch/images/CE3.png)

**Evaluation Methodology:**

We created a validation dataset of size 10,000, using the same DataClass as train set.

We measured and compared the Total loss on training set vs Validation set 

and Accuracy on Train vs Validation set.

We trained the model till the accuracy  

**Our Results:**

These are our results after training:

Training logs:

![Logs](https://github.com/hassiahk/EVA6-Phase1-Assignments/blob/main/Session3-Pytorch/images/Selection_191.png)

The best validation accuracy: 88.19 %

Loss curve:

![Loss](https://github.com/hassiahk/EVA6-Phase1-Assignments/blob/main/Session3-Pytorch/images/Selection_190.png)

Accuracy curve:

![Accuracy](https://github.com/hassiahk/EVA6-Phase1-Assignments/blob/main/Session3-Pytorch/images/Selection_192.png)

All the training was done on **GPU,** by pushing our model and tensors to device = cdua.