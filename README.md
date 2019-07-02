# Domain agnostic learning with disentangled representations
<img src='img/overview.png'>

PyTorch implementation for **Domain agnostic learning with disentangled representations**.


## Citation
If you use this code for your research, please cite our [paper](http://proceedings.mlr.press/v97/peng19b/peng19b.pdf):

```
@InProceedings{pmlr-v97-peng19b,
  title = 	 {Domain Agnostic Learning with Disentangled Representations},
  author = 	 {Peng, Xingchao and Huang, Zijun and Sun, Ximeng and Saenko, Kate},
  booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
  pages = 	 {5102--5112},
  year = 	 {2019},
  editor = 	 {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
  volume = 	 {97},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Long Beach, California, USA},
  month = 	 {09--15 Jun},
  publisher = 	 {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v97/peng19b/peng19b.pdf},
  url = 	 {http://proceedings.mlr.press/v97/peng19b.html},
  abstract = 	 {Unsupervised model transfer has the potential to greatly improve the generalizability of deep models to novel domains. Yet the current literature assumes that the separation of target data into distinct domains is known a priori. In this paper, we propose the task of Domain-Agnostic Learning (DAL): How to transfer knowledge from a labeled source domain to unlabeled data from arbitrary target domains? To tackle this problem, we devise a novel Deep Adversarial Disentangled Autoencoder (DADA) capable of disentangling domain-specific features from class identity. We demonstrate experimentally that when the target domain labels are unknown, DADA leads to state-of-the-art performance on several image classification datasets.}
}
```
