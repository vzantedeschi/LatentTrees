# Learning Binary Trees by Argmin Differentiation
Code source of ICML 2021 paper [Learning Binary Trees by Argmin Differentiation](https://arxiv.org/abs/2010.04627).

### Dependencies

Install PyTorch, following the [guidelines](https://pytorch.org/get-started/locally/).

On Ubuntu16.04+, make sure you have GLIBCXX_3.4.22 support via libstdc++.so.6:

```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-4.9
sudo apt-get upgrade libstdc++6
sudo apt-get dist-upgrade
```

Setting up the cpp extensions requires **gcc-9** or above:

```bash
sudo apt install gcc-9
sudo apt install g++-9
```

Plotting with Networkx requires the following libraries:

```bash
sudo apt-get install python3-dev graphviz libgraphviz-dev pkg-config
```
### Setup
```bash
pip3 install -r requirements.txt
CXX=gcc python3 setup.py build_ext --inplace
```

### Train on toy datasets
```bash
python3 fit_toyset.py
```

Default configuration is stored in 'config/default-xor.yaml'. You can edit directly the config file or change values from the command line, e.g. as follows: 
```bash
python3 fit_toyset.py dataset.N=1000 model.SPLIT=linear
```
See [Hydra](https://hydra.cc/docs/intro/) for a tutorial.

### Citation

``` bibtex
  @article{zantedeschi2021learning,
    title={Learning Binary Trees by Argmin Differentiation},
    author={Zantedeschi, Valentina and Kusner, Matt J and Niculae, Vlad},
    journal={ICML},
    year={2021}
  }
```

