import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar y preprocesar el conjunto de datos MNIST
print("Cargando el conjunto de datos MNIST...")
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
print("Datos cargados correctamente.")

# Definir el Stacked Autoencoder
class StackedAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, hidden_dim):
        super(StackedAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

# Instanciar el modelo
input_dim = 28 * 28  # 784 píxeles
encoding_dim = 128
hidden_dim = 64
model = StackedAutoencoder(input_dim, encoding_dim, hidden_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenar el Autoencoder
print("Entrenando el Autoencoder...")
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    for images, _ in train_loader:
        images = images.to(device)
        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
print("Entrenamiento finalizado.")

# Extraer representaciones comprimidas para entrenamiento y prueba
print("Extrayendo representaciones comprimidas...")
def extract_features(model, dataloader):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            _, encoded = model(images)
            features.append(encoded.cpu().numpy())
            labels.append(targets.numpy())
    return np.vstack(features), np.hstack(labels)

encoded_train, y_train = extract_features(model, train_loader)
encoded_test, y_test = extract_features(model, test_loader)
print("Representaciones comprimidas obtenidas.")

# Clasificador con la representación comprimida
print("Entrenando clasificador Random Forest con las representaciones comprimidas...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=2, n_jobs=-1)
clf.fit(encoded_train, y_train)
y_pred = clf.predict(encoded_test)
print("Clasificación completada.")

# Evaluar el rendimiento del clasificador
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy del clasificador con Stacked Autoencoder: {accuracy:.4f}')

''' LOG
Cargando el conjunto de datos MNIST...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9.91M/9.91M [00:01<00:00, 6.20MB/s] 
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 28.9k/28.9k [00:00<00:00, 265kB/s] 
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.65M/1.65M [00:00<00:00, 2.97MB/s] 
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4.54k/4.54k [00:00<00:00, 771kB/s] 
Datos cargados correctamente.
Entrenando el Autoencoder...
Epoch [1/10], Loss: 0.0409
Epoch [2/10], Loss: 0.0281
Epoch [3/10], Loss: 0.0204
Epoch [4/10], Loss: 0.0182
Epoch [5/10], Loss: 0.0162
Epoch [6/10], Loss: 0.0142
Epoch [7/10], Loss: 0.0125
Epoch [8/10], Loss: 0.0121
Epoch [9/10], Loss: 0.0119
Epoch [10/10], Loss: 0.0112
Entrenamiento finalizado.
Extrayendo representaciones comprimidas...
Representaciones comprimidas obtenidas.
Entrenando clasificador Random Forest con las representaciones comprimidas...
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 12 concurrent workers.
building tree 1 of 100
building tree 2 of 100
building tree 4 of 100
building tree 3 of 100
...

building tree 29 of 100
[Parallel(n_jobs=-1)]: Done  17 tasks      | elapsed:    4.7s
building tree 30 of 100
...
building tree 100 of 100
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:   20.7s finished
[Parallel(n_jobs=12)]: Using backend ThreadingBackend with 12 concurrent workers.
[Parallel(n_jobs=12)]: Done  17 tasks      | elapsed:    0.0s
[Parallel(n_jobs=12)]: Done 100 out of 100 | elapsed:    0.1s finished
Clasificación completada.
Accuracy del clasificador con Stacked Autoencoder: 0.947
'''