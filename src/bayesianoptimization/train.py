from bayesianoptimization.model import Model
from bayesianoptimization.data import MyDataset
from skopt.space import Integer, Real


search_space = [
    Integer(2, 5, 10, 20, name = 'depth'),
    Integer(3, 5, 7, 9, name = "kernel_size"),
    Real(0.0, 0.1, 0.3, 0.5, name = "dropout_rate")
]






def train():
    dataset = MyDataset("data/raw")
    model = Model()
    # add rest of your training code here

if __name__ == "__main__":
    train()
