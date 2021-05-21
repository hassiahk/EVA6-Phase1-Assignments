Authors:

1. Haswants Aekula
2. Shreeyash Pawar
3. Smita 
4. Manjeera

We have designed a neural architecture such that:

We have designed a neural architecture such that:**Input** : an MNIST image and Random number 

**Outputs:** MNIST Digit and sum of mnist digit + random number.

![mnist-statement](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a14e7242-ca15-4545-a240-c26c60f9d9a5/Untitled.png)

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

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5a6cee01-0550-4b45-bb3e-5f5bded1c10a/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5a6cee01-0550-4b45-bb3e-5f5bded1c10a/Untitled.png)

Here: out → MNIST digit, sum_out → MNIST digit + out

Model architecture:

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/afb9bd98-4003-464a-a945-07e8547ae2c2/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/afb9bd98-4003-464a-a945-07e8547ae2c2/Untitled.png)

**Loss Function:**

We used 2 losses , one for each ouput , and averaged them to get total loss.

The individual losses used were "cross_entropy" loss.

Since this is classification task, where we had to classify numbers for both the outputs, and hence use entropy loss as:

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8bc9fc37-b049-47ac-99a4-1293463b1150/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8bc9fc37-b049-47ac-99a4-1293463b1150/Untitled.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/581077aa-48eb-42ba-8798-927e746d8bae/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/581077aa-48eb-42ba-8798-927e746d8bae/Untitled.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/212135fe-5dfa-4c61-b15c-c97d7ad65632/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/212135fe-5dfa-4c61-b15c-c97d7ad65632/Untitled.png)

**Evaluation Methodology:**

We created a validation dataset of size 10,000, using the same DataClass as train set.

We measured and compared the Total loss on training set vs Validation set 

and Accuracy on Train vs Validation set.

We trained the model till the accuracy  

**Our Results:**

These are our results after training:

Training logs:

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5f21ed2a-a97e-4d44-b2af-3563c458db51/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5f21ed2a-a97e-4d44-b2af-3563c458db51/Untitled.png)

The best validation accuracy: 88.19 %

Loss curve:

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fe1687b9-b241-445f-b94f-3feed3979718/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/fe1687b9-b241-445f-b94f-3feed3979718/Untitled.png)

Accuracy curve:

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5f99c0e0-d6dc-47c0-9ae3-db0fba114bb3/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5f99c0e0-d6dc-47c0-9ae3-db0fba114bb3/Untitled.png)

All the training was done on **GPU,** by pushing our model and tensors to device = cdua.