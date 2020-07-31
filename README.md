# Latent Tree Learning

### Dependencies

On Ubuntu16.04+, make sure you have GLIBCXX_3.4.22 support via libstdc++.so.6:

```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-4.9
sudo apt-get upgrade libstdc++6
sudo apt-get dist-upgrade
```

Setting up the cpp extensions requires at least **gcc-9**:

```bash
sudo apt install gcc-9
sudo apt install g++-9
```

### Setup
```python
CXX=gcc python3 setup.py build_ext --inplace
```