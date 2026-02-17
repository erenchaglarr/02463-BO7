from bayesianoptimization.model import CNNModel
from bayesianoptimization.data import MyDataset
from skopt.space import Integer, Real
from skopt import gp_minimize, dummy_minimize


search_space = [
    Integer(2, 20, name = 'depth'),
    Integer(3, 9, name = "kernel_size"),
    Real(0.0, 0.5, name = "dropout_rate")
]

def objective(params):
    depth, kernel_size, dropout_rate = params

    model = CNNModel(
        depth=depth,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate
    )

    validation_score = "INDSÆT TRÆNINGSFUNKTIONEN HER"

    return validation_score

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


def train():
    dataset = MyDataset("data/omniglot-py/images_background")
    model = CNNModel()
    # add rest of your training code here

if __name__ == "__main__":
    train()
