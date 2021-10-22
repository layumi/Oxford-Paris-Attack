# CNN Image Retrieval in PyTorch: Training and evaluating CNNs for Image Retrieval in PyTorch

The code is based on the https://github.com/filipradenovic/cnnimageretrieval-pytorch .

We added our attacking method ODFA (https://arxiv.org/abs/1809.02681).  The performance drops from 88.2% to 2.24% on Oxford.

The results attacking other datasets can be found at our main repo https://github.com/layumi/A_reID .

### Regular Testing

We provide the pretrained networks trained using the same parameters as in our TPAMI 2018 paper, with precomputed whitening. To evaluate them run:
```
python3 -m cirtorch.examples.test --gpu-id '2' --network-path 'retrievalSfM120k-resnet101-gem' --datasets 'oxford5k,paris6k,roxford5k,rparis6k' --whitening 'retrieval-SfM-120k' --multiscale '[1]'
```

The table below shows the performance comparison of networks trained with this framework and the networks used in the paper which were trained with our [CNN Image Retrieval in MatConvNet](https://github.com/filipradenovic/cnnimageretrieval):

| Model | Oxford | Paris | ROxf (M) | RPar (M) | ROxf (H) | RPar (H) |
|:------|:------:|:------:|:------:|:------:|:------:|:------:|
| ResNet101-GeM (PyTorch) Multiple Scale | 88.2 | 92.5 | 65.4 | 76.7 | 40.1 | 55.2 |
| ResNet101-GeM (PyTorch) Single Scale   | 86.24 |  90.66  | - | - | - | - |
| VGG16-GeM (PyTorch) Single Scale   | 85.24 |  86.28  | - | - | - | - |

### Testing Adversarial Query
We add one function `cirtorch.network.extract_vectors_aq` and one py file `test_aq.py`.

single-scale
```
python3 -m cirtorch.examples.test_aq --gpu-id '2' --network-path 'retrievalSfM120k-resnet101-gem' --datasets 'oxford5k,paris6k' --whitening 'retrieval-SfM-120k' --multiscale '[1]'
```

multi-scale
```
python3 -m cirtorch.examples.test_aq --gpu-id '2' --network-path 'retrievalSfM120k-resnet101-gem' --datasets 'oxford5k,paris6k' --whitening 'retrieval-SfM-120k' --multiscale '[1, 1/2**(1/2), 1/2]'
```

| Model | Oxford | Paris | ROxf (M) | RPar (M) | ROxf (H) | RPar (H) |
|:------|:------:|:------:|:------:|:------:|:------:|:------:|
| ResNet101-GeM (PyTorch) Multiple Scale | 2.24 (1.82) | 4.78 (3.64) | - | - | - | - |
| ResNet101-GeM (PyTorch) Single Scale  | 0.80 | 3.00 | - | - | - | - |
| VGG16-GeM (PyTorch) Single Scale   | 0.64 | 1.85 | - | - | - | - |

## Related publications

### Training (fine-tuning) convolutional neural networks 
```
@article{RTC18,
 title = {Fine-tuning {CNN} Image Retrieval with No Human Annotation},
 author = {Radenovi{\'c}, F. and Tolias, G. and Chum, O.}
 journal = {TPAMI},
 year = {2018}
}
```
```
@inproceedings{RTC16,
 title = {{CNN} Image Retrieval Learns from {BoW}: Unsupervised Fine-Tuning with Hard Examples},
 author = {Radenovi{\'c}, F. and Tolias, G. and Chum, O.},
 booktitle = {ECCV},
 year = {2016}
}
```

### Revisited benchmarks for Oxford and Paris ('roxford5k' and 'rparis6k')
```
@inproceedings{RITAC18,
 author = {Radenovi{\'c}, F. and Iscen, A. and Tolias, G. and Avrithis, Y. and Chum, O.},
 title = {Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking},
 booktitle = {CVPR},
 year = {2018}
}
```

## Versions

### [master](https://github.com/filipradenovic/cnnimageretrieval-pytorch/tree/master) (development)

- Migrated code to PyTorch 1.0.0, removed Variable, added torch.no_grad for more speed and less memory at evaluation
- Added rigid grid regional pooling that can be combined with any global pooling method (R-MAC, R-SPoC, R-GeM)
- Added PowerLaw normalization layer
- Added multi-scale testing with any given set of scales, in example test script
- Fix related to precision errors of covariance matrix estimation during whitening learning
- Fixed minor bugs

### [v1.0](https://github.com/filipradenovic/cnnimageretrieval-pytorch/tree/v1.0) (09 Jul 2018)

- First public version
- Compatible with PyTorch 0.3.0
