# Collaborative Defense-GAN for Protecting Adversarial Attacks on Classification System
Publication: https://www.sciencedirect.com/science/article/abs/pii/S0957417422019753  
Defense an adversarial examples with discover GAN 2 domain mapping.

`pip install -r requirements.txt`

Since lasted version of advertorch is not avaiable on PYPI you can download by following this

    git clone https://github.com/BorealisAI/advertorch
    cd advertorch
    pip install -e .


To train model you have to train classifier first.

`python train_classifier.py --data mnist --arch B --n_epochs 2`

Train Defense model.

`python train.py --data mnist --arch B --clf_model ./saved_model/mnist_B.pth --n_epochs 2 --batch_size 256`

Detail of script argument are in `--help` command ex `python train.py --help`
