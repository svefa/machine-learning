import sys
import os
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt


# Pandas: Reading the data
df = pd.read_csv("./data/used_cars.csv")

# Pandas: Preparing the data
age = df["model_year"].max() - df["model_year"]

milage = df["milage"]
milage = milage.str.replace(",", "")
milage = milage.str.replace(" mi.", "")
milage = milage.astype(int)

accident_free = df["accident"] == "None reported"
accident_free = accident_free.astype(int)

price = df["price"]
price = price.str.replace("$", "")
price = price.str.replace(",", "")
price = price.astype(int)

# Torch: Creating X and y data (as tensors)
X = torch.column_stack([
    torch.tensor(accident_free, dtype=torch.float32),
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(milage, dtype=torch.float32)
])
# Normalize
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

y = torch.tensor(price, dtype=torch.float32)\
    .reshape((-1, 1))
# Normalize
y_mean = y.mean()
y_std = y.std()
y = (y - y_mean) / y_std
# sys.exit()


model = nn.Linear(3, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

model.train()

losses = []
for i in range(0, 10000):
    # Training pass
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if i % 1000 == 0: 
        print(loss.item())

plt.plot(losses)
# plt.show()

# Saving and Loading Model
if not os.path.isdir("./model"):
    os.mkdir("./model")
torch.save(X_mean, "./model/X_mean.pt")
torch.save(X_std, "./model/X_std.pt")
torch.save(model.state_dict(), "./model/model.pt")

model = nn.Linear(3, 1)
model.load_state_dict(
    torch.load("./model/model.pt", weights_only=True)
    )
X_mean = torch.load("./model/X_mean.pt", weights_only=True)
X_std = torch.load("./model/X_std.pt", weights_only=True)

# Prediction
model.eval()
X_data = torch.tensor([
    [ 0, 5, 10000],
    [ 0, 2, 10000],
    [ 0, 5, 20000]
], dtype=torch.float32)

prediction = model((X_data - X_mean) / X_std)
print(prediction * y_std + y_mean)

X_data_accident = torch.tensor([
    [ 1, 5, 10000],
    [ 1, 2, 10000],
    [ 1, 5, 20000]
], dtype=torch.float32)

prediction_accident = model((X_data_accident - X_mean) / X_std)
print(prediction_accident * y_std + y_mean)