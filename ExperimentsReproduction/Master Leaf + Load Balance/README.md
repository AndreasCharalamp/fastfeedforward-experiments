# How to use:

## fff.py 
fff+masterleaf.py includes the definition of the FFF class with the implementation of Master Leaf. Building upon the work of Peter Belcak and Roger Wattenhofer, we have merely added a calculation of f_i s and P_i s needed 
for the load balancing term in Loss fucntion. Furthermore, we added the implemetation of Master Leaf.

## main.py
main.py is the file that excecutes a training procedure. 
### How to use

1. MNIST
```sh
python main.py  MNIST --batch-size=256 --leaf-width=8 --depth=1 --balance-epochs=300 --hard-epochs=300 --runs=10
```

### Results
main.py will store results in a folder named: (DATASET)_l(leaf_width)_d(depth)/test_i. 
e.g.
If we excecute the following comands in succession:
```sh
python main.py  MNIST --batch-size=256 --leaf-width=8 --depth=1 --balance-epochs=300 --hard-epochs=300 --runs=10
```
```sh
python main.py  MNIST --batch-size=256 --leaf-width=8 --depth=1 --balance-epochs=300 --hard-epochs=300 --runs=10
```
```sh
python main.py  MNIST --batch-size=256 --leaf-width=8 --depth=4 --balance-epochs=300 --hard-epochs=300 --runs=10
```
Our directory will be the following:
```bash
├── Load Balance
│   ├── MNIST_l8_d1
│   │   ├── test_1
│   │   └── test_2
│   ├── MNIST_l8_d4
│   │   └── test_1
│   ├── fff.py
└── └── main.py
```
Where MNIST_l8_d1/test_i is created for the excecution of the ith time we excecute main.py with DATASET=MNIST, leaf-width=8 and depth = 1.
Each test folder will contain the parameters of the 10 models trained in the format of a dictionary and a .txt with the metrics of each model for all epochs run.

