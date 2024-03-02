import json
import itertools
import numpy as np

def generate_configurations(params, constants, search_type='grid', num_random_samples=10):
    """
    Generates configurations based on the specified search type.
    
    :param params: Parameters for the search, excluding constants.
    :param constants: Constant parameters across all runs.
    :param search_type: 'grid' or 'random' to specify the type of search.
    :param num_random_samples: Number of configurations for random search.
    :return: List of configurations.
    """
    configurations = []
    
    if search_type == 'grid':
        keys, ranges = zip(*[(k, np.arange(v['min'], v['max'] + v['step'], v['step'])) for k, v in params.items()])
        for values in itertools.product(*ranges):
            # Convert Numpy data types to Python native types for JSON serialization
            config = {key: value.item() if isinstance(value, np.generic) else value for key, value in zip(keys, values)}
            configurations.append(config)
    elif search_type == 'random':
        for _ in range(num_random_samples):
            configuration = {k: round(np.random.uniform(v['min'], v['max']), 2) for k, v in params.items()}
            # Ensure all values are native Python types
            configuration = {k: v.item() if isinstance(v, np.generic) else v for k, v in configuration.items()}
            configurations.append(configuration)
    else:
        raise ValueError("Invalid search type. Choose 'grid' or 'random'.")

    # Add constant parameters to each configuration
    for config in configurations:
        config.update(constants)
    
    return configurations

def save_configurations(configurations, filename="configurations.json"):
    """
    Saves configurations to a JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(configurations, f, indent=4)

if __name__ == "__main__":
    params = {
        '--alpha': {'min': 0, 'max': 15, 'step': 1},
    }
    # Random example
    # params = {
    #     'ALPHA': {'min': 5, 'max': 20},
    # }
    constants = {
        '--recompute-mean-std': True,
        '--show-binary-frames': True,
    }
    search_type = 'grid'  # Choose either 'grid' or 'random'
    num_random_samples = 5  # Relevant for random search only
    
    configurations = generate_configurations(params, constants, search_type, num_random_samples)
    save_configurations(configurations)
