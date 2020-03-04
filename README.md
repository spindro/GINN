# Graph Imputation Neural Networks (GINN)

This is the companion code for the paper:
[Missing Data Imputation with Adversarially-trained Graph Convolutional Networks](https://arxiv.org/abs/1905.01907), *arXiv:1905.01907*, 2019.

### Imputing missing data with graph neural networks

We perform imputation of missing data in a generic dataset by (a) building a graph of similarities between examples, and (b) running an autoencoder with graph convolutions [1] on top of that.

![Generic schematics of our imputation method](https://github.com/spindro/GINN/blob/master/download.png)

### Organization of the code

All the code for the models described in the paper can be found in *ginn/core.py* and *ginn/models.py*. Examples of use with accompanying notebooks are in *examples*.

### References

[1] Kipf, T.N. and Welling, M., 2016. **Semi-supervised classification with graph convolutional networks**. arXiv preprint arXiv:1609.02907.

### Cite

Please cite [our paper](https://arxiv.org/abs/1905.01907) if you use this code in your own work:

```
@article{spinelli2019ginn,
  title={Missing Data Imputation with Adversarially-trained Graph Convolutional Networks},
  author={Spinelli, Indro and Scardapane, Simone and Aurelio, Uncini},
  journal={arXiv preprint arXiv:1905.01907},
  year={2019}
}
```