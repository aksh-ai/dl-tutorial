import torch
import torch.nn as nn
import torch.nn.functional as F 

class ANN(nn.Module):
    def __init__(self, in_feat=4, hidden_units_1=8, hidden_units_2=4, out_feat=3):
        super().__init__()
        self.input_layer = nn.Linear(in_feat, hidden_units_1)
        self.hidden_layer = nn.Linear(hidden_units_1, hidden_units_2)
        self.output_layer = nn.Linear(hidden_units_2, out_feat)
    
    def forward(self, X):
        x = F.leaky_relu(self.input_layer(X), negative_slope=0.3)
        x = F.leaky_relu(self.hidden_layer(x), negative_slope=0.3)
        x = self.output_layer(x)
        return x

model = ANN()

model.load_state_dict(torch.load('models/iris_ANN.pt'))

model.eval()

features = ['Sepal Width (cm)', 'Sepal Length (cm)', 'Petal Width (cm)', 'Petal Height (cm)']
labels = ['Iris setosa','Iris virginica','Iris versicolor','Unknown']

in_feat = []

for feature in features:
	val = input('Enter {} : '.format(feature))
	in_feat.append(float(val))

in_feat = torch.FloatTensor(in_feat)

with torch.no_grad():
    print(f'\n{labels[model(in_feat).argmax()]}\n')