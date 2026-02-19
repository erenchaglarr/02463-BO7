from bayesianoptimization.model import CNNModel
from bayesianoptimization.data import MyDataset
from skopt.space import Integer, Real
from skopt import gp_minimize, dummy_minimize
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

search_space = [
    Integer(2, 20, name = 'depth'),
    Integer(3, 9, name = "kernel_size"),
    Real(0.0, 0.5, name = "dropout_rate")
]

def objective(params):
    depth, kernel_size, dropout_rate = params
    validation_score = train(depth, kernel_size, dropout_rate)
    return -validation_score



def train(depth, kernel_size, dropout_rate):
    torch.manual_seed(42)
    np.random.seed(42)
    dataset = MyDataset("data/omniglot-py/images_background")
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset) - int(0.8*len(dataset))], generator=generator)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)


    model = CNNModel(
        depth=depth,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 5

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = correct / total

    return accuracy

if __name__ == "__main__":
    results_ei = gp_minimize(
    func=objective,
    dimensions=search_space,
    n_calls=20,
    n_random_starts=5,
    acq_func='EI', 
    random_state=42
    )

    results_pi = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=20,
        n_random_starts=5,
        acq_func='PI', 
        random_state=42
    )

    results_random = dummy_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=20,
        random_state=42
    )
    print("Best EI:", -results_ei.fun)
    print("Best PI:", -results_pi.fun)
    print("Best Random:", -results_random.fun)
