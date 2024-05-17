from fff import FFF
import argparse
from tqdm import tqdm
import torch
import os
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['MNIST', 'FashionMNIST'])
parser.add_argument('-b', '--batch-size', type=int, default=2048)
parser.add_argument('-i', '--input-width', type=int, default=784)
parser.add_argument('-l', '--leaf-width', type=int, default=1)
parser.add_argument('-hp', '--harden-param', type=int, default=1)
parser.add_argument('-ap', '--balance-param', type=int, default=1)
parser.add_argument('--depth', type=int, default=11)
parser.add_argument('-be', '--balance-epochs', type=int, default=100)
parser.add_argument('-he', '--hard-epochs', type=int, default=100)
parser.add_argument('-w', '--hidden-width', type=int, default=128)
parser.add_argument('-r', '--runs', type=int, default=1)
options = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
dataset = options.dataset
batch_size = options.batch_size
entropy_effect = options.harden_param
n_epochs = options.balance_epochs
h_epochs = options.hard_epochs
alpha = options.balance_param
leaf_width = options.leaf_width
depth = options.depth
runs = options.runs
activation = nn.ReLU()
leaf_dropout = 0.0
region_leak = 0.0
criterion = nn.CrossEntropyLoss()
model2 = FFF(input_width=784, leaf_width=leaf_width, output_width=10, depth=depth, activation=activation, dropout=leaf_dropout, region_leak=region_leak, usage_mode = 'soft')
optimizer = torch.optim.AdamW(model2.parameters(), lr=0.001)

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory+"_1")
        print(f"Directory '{directory}'_1 created.")
        return directory
    else:
        count = 2
        while True:
            new_directory = f"{directory}_{count}"
            if not os.path.exists(new_directory):
                os.makedirs(new_directory)
                print(f"Directory '{new_directory}' created.")
                return new_directory
            count += 1


folder_path = str(dataset) + "_l"+ str(leaf_width) + "_d" + str(depth)
folder_path = folder_path + "/test"
folder_path = create_directory(folder_path)
# Load dataset
from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
data_dic = {
   "MNIST": [datasets.MNIST('data', download=True, train=True, transform=transform), datasets.MNIST('data', download=True, train=False, transform=transform)],
   "FashionMNIST": [datasets.FashionMNIST('data', download=True, train=True, transform=transform), datasets.FashionMNIST('data', download=True, train=False, transform=transform)]
}

dataloader_training = torch.utils.data.DataLoader(data_dic[dataset][0], batch_size=batch_size, shuffle=True)
dataloader_testing = torch.utils.data.DataLoader(data_dic[dataset][1], batch_size=batch_size, shuffle=True)
dataloader_training2 = torch.utils.data.DataLoader(data_dic[dataset][0], batch_size=60000, shuffle=True)
dataloader_testing2 = torch.utils.data.DataLoader(data_dic[dataset][1], batch_size=10000, shuffle=False)


def balance_step(a,h):
   for batch_images, batch_labels in tqdm(dataloader_training):
      optimizer.zero_grad()
      batch_images = batch_images.to(device)
      batch_labels = batch_labels.to(device)

      output = model2(batch_images.view(-1, 784), return_entropies=True)
      node_entropy_mean = output[1].mean()
      loss = criterion(output[0], batch_labels) + h * node_entropy_mean + a*(2**depth)*(1/batch_size)*(1/batch_size)*output[2]

      loss.backward()
      optimizer.step()
   return

def test(dataloader):
   for batch_images, batch_labels in tqdm(dataloader):
      batch_images = batch_images.to(device)
      batch_labels = batch_labels.to(device)
      output = model2(batch_images.view(-1, 784))
      #print(output)
      if type(output) is tuple:
         accuracy = (output[0].argmax(dim=1) == batch_labels).detach().float().mean()
      else:
         accuracy = (output.argmax(dim=1) == batch_labels).detach().float().mean()
      return accuracy.item()


def hard_step(h):
   for batch_images, batch_labels in tqdm(dataloader_training):
      optimizer.zero_grad()
      batch_images = batch_images.to(device)
      batch_labels = batch_labels.to(device)

      output = model2(batch_images.view(-1, 784), return_entropies=True)
      node_entropy_mean = output[1].mean()
      loss = criterion(output[0], batch_labels) + h * node_entropy_mean 
      loss.backward()
      optimizer.step()
   return 


def load_model(pth):
  checkpoint = torch.load(pth)
  model2.node_weights = torch.nn.Parameter(checkpoint['node_weights'])
  model2.node_biases = torch.nn.Parameter(checkpoint['node_biases'])
  model2.w1s = torch.nn.Parameter(checkpoint['w1s'])
  model2.b1s = torch.nn.Parameter(checkpoint['b1s'])
  model2.w2s = torch.nn.Parameter(checkpoint['w2s'])
  model2.b2s = torch.nn.Parameter(checkpoint['b2s'])

f = open(folder_path + "/test_results_a" + str(alpha) + "_h" +str(entropy_effect) + ".txt", "w")
print(options, file = f)
for i in range(runs):
  # setup the FFF model
  model2 = FFF(input_width=784, leaf_width=leaf_width, output_width=10, depth=depth, activation=activation, dropout=leaf_dropout, region_leak=region_leak, usage_mode = 'soft')
  model2.to(device)
  file_path = "model_" +str(i)
  model2.new_impl = True
  torch.save(model2.state_dict(), folder_path + "/" + file_path)
  model2.eval()
  best_acc = test(dataloader_testing2)
  counter = 0
  lr=0.001
  print(f'Model number {i+1}:',file = f)
  optimizer = torch.optim.AdamW(model2.parameters(), lr=lr)
  criterion = nn.CrossEntropyLoss()
  for epoch in range(n_epochs):
    print("Epoch ", epoch + 1, file = f)
    print("Epoch ", epoch + 1)
    model2.train()
    balance_step(alpha, entropy_effect)
    

    # test the model
    model2.eval()
    test_acc = test(dataloader_testing2)
    print("test acc: ",test_acc , file = f)
    if(best_acc < test_acc):
      torch.save(model2.state_dict(), folder_path + "/" + file_path)
      print("Test accuracy is: ", test_acc)
      best_acc = test_acc
      counter = 0
      if (lr > 0.001):
         lr = 0.001
         optimizer = torch.optim.AdamW(model2.parameters(), lr=lr)
    elif (counter < 5 and lr < 0.003  ):
       counter +=1
    elif (best_acc<0.90 and lr < 0.003):
       counter = 0
       lr = 0.003
       optimizer = torch.optim.AdamW(model2.parameters(), lr=lr)
    else: 
       counter+=1
       if counter >  1000:
          lr = 0.001
          break
  
  #Hardening the model
  #file_path = file_path + "_hardened"
  load_model(folder_path + "/" + file_path)
  file_path = file_path + "_hardened"
  torch.save(model2.state_dict(), folder_path + "/" + file_path)
  for epoch in range(h_epochs):
    print("Epoch ", epoch + n_epochs + 1, file = f)
    print("Epoch ", epoch + n_epochs + 1)
    model2.train()
    hard_step(3)
    # test the model
    model2.eval()
    test_acc = test(dataloader_testing2)
    print("test acc: ",test_acc , file = f)
    if(best_acc < test_acc):
      torch.save(model2.state_dict(), folder_path + "/" + file_path)
      print("Test accuracy is: ", test_acc)
      best_acc = test_acc
      counter = 0
      if (lr > 0.001):
         lr = 0.001
         optimizer = torch.optim.AdamW(model2.parameters(), lr=lr)
    elif (counter < 5 ):
       counter +=1
    elif (best_acc<0.90):
       counter = 0
       lr = 0.003
       optimizer = torch.optim.AdamW(model2.parameters(), lr=lr)

  
  load_model(folder_path + "/" + file_path)
  model2.train()

  for batch_images, batch_labels in tqdm(dataloader_testing2):
      batch_images = batch_images.to(device)
      batch_labels = batch_labels.to(device)
      output = model2(batch_images.view(-1, 784))
      accuracy = (output.argmax(dim=1) == batch_labels).detach().float().mean()
  print("Leaf usage is: ", (model2.f!=0).float().sum().item(), file = f)
  train_acc = test(dataloader_training2)
  print("train acc: ",train_acc, file = f)
  print("Best acc is: ", best_acc, file = f)
  
f.close()