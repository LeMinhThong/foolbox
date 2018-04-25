from numpy import linalg as LA
import torch, foolbox, random
import numpy as np
from models import MNIST, CIFAR10, IMAGENET, SimpleMNIST, load_mnist_data, load_cifar10_data, imagenettest, load_model
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '5'

train_loader, test_loader, train_dataset, test_dataset = load_cifar10_data()

net = CIFAR10()
if torch.cuda.is_available():
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=[0])
        
load_model(net, 'models/cifar10_gpu.pt')
#load_model(net, 'models/cifar10_cpu.pt')
net.eval()

model = net.module if torch.cuda.is_available() else net

fmodel = foolbox.models.PyTorchModel(model, bounds=[0,1], num_classes=10)
attack = foolbox.attacks.BoundaryAttack(fmodel)

random.seed(0)
avg_dis = 0
avg_calls = 0
num_attacks = 2

samples = [6411, 4360, 7753, 7413, 684, 3343, 6785, 7079, 2263]
samples = [4360]
for idx in samples:
        image, label = test_dataset[idx]
        image = image.numpy()
        print("\n\n\n\n======== Image %d =========" % idx)
        #adv = attack(image,label,iterations=5000, verbose=False, unpack=False, log_every_n_steps=1000)
        adv = attack(image,label,iterations=5000, verbose=True, unpack=False, threaded_gen=1, threaded_rnd=1)
        avg_dis += LA.norm(adv.image-image)
        avg_calls += adv._total_prediction_calls
        print("Norm and queries of {} is {} and {}".format(idx, LA.norm(adv.image-image), adv._total_prediction_calls))

print("average distortion and queries is {} and {}".format(avg_dis/len(samples), avg_calls/len(samples)))