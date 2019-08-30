# DAL with disentangled representations
<img src='img/overview.png'>

PyTorch implementation for **Domain agnostic learning with disentangled representations** (ICML2019 Long Oral). This repository contains some code from [Maximum Classifier Discrepancy for Domain Adaptation](https://github.com/mil-tokyo/MCD_DA). If you find this repository useful for you, please also consider cite the MCD paper!

The code has been tested on Python 3.6+PyTorch 0.3 and Python 3.6+PyTorch 0.41. To run the training and testing code, use the following script:

```
python main.py --source=mnist --recordfolder=agnostic_disentangle --gpu=0
```

The poster of this paper can be found with the link: [poster](https://cs-people.bu.edu/xpeng/pdfs/DAL_ICML2019_Poster.pdf)

The **Oral** presentation of this paper in ICML2019 can be found with the link: [Oral Presentation](https://slideslive.com/38917425/transfer-and-multitask-learning)

## Citation
If you use this code for your research, please cite our [paper](http://proceedings.mlr.press/v97/peng19b/peng19b.pdf):

```
@inproceedings{Peng2019DomainAL,
  title={Domain Agnostic Learning with Disentangled Representations},
  author={Xingchao Peng and Zijun Huang and Ximeng Sun and Kate Saenko},
  booktitle={ICML},
  year={2019}
}
```
