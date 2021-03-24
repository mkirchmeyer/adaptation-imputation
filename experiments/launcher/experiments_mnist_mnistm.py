activate_mse = 1
activate_adaptation_imp = 1
activate_adaptation_d1 = 1
weight_d2 = 1.0
weight_mse = 1.0
refinement = 1
n_epochs_refinement = 10
lambda_regul = [0.01]
lambda_regul_s = [0.01]
threshold_value = [0.95]

compute_variance = False
random_seed = [1985] if not compute_variance else [1985, 2184, 51, 12, 465]


class DannMNISTMNISTM(object):
    MAX_NB_PROCESSES = 3
    DEBUG = False
    BINARY = "experiments/launcher/digits_binary.py"
    GRID = {
        "-mode": ["dann"],
        "-upper_bound": [1],
        "-adaptive_lr": [1],
        "-is_balanced": [1],
        "-source": ["MNIST"],
        "-target": ["MNISTM"],
        "-n_epochs": [100],
        "-batch_size": [128],
        "-epoch_to_start_align": [11],
        "-initialize_model": [1],
        "-init_batch_size": [32],
        "-refinement": [refinement],
        "-n_epochs_refinement": [n_epochs_refinement],
        "-lambda_regul": lambda_regul,
        "-lambda_regul_s": lambda_regul_s,
        "-threshold_value": threshold_value,
        "-init_lr": [10 ** -2],
        "-random_seed": random_seed
    }


class DannIgnoreMNISTMNISTM(object):
    MAX_NB_PROCESSES = 3
    DEBUG = False
    BINARY = "experiments/launcher/digits_binary.py"
    GRID = {
        "-mode": ["dann"],
        "-upper_bound": [1],
        "-adaptive_lr": [1],
        "-is_balanced": [1],
        "-source": ["MNIST"],
        "-target": ["MNISTM"],
        "-n_epochs": [100],
        "-batch_size": [128],
        "-epoch_to_start_align": [11],
        "-initialize_model": [1],
        "-init_batch_size": [32],
        "-init_lr": [10 ** -2],
        "-adapt_only_first": [1],
        "-refinement": [refinement],
        "-n_epochs_refinement": [n_epochs_refinement],
        "-lambda_regul": lambda_regul,
        "-threshold_value": threshold_value,
        "-random_seed": random_seed
    }


class DannZeroImputMNISTMNISTM(object):
    MAX_NB_PROCESSES = 3
    DEBUG = False
    BINARY = "experiments/launcher/digits_binary.py"
    GRID = {
        "-mode": ["dann"],
        "-upper_bound": [0],
        "-adaptive_lr": [1],
        "-is_balanced": [1],
        "-source": ["MNIST"],
        "-target": ["MNISTM"],
        "-n_epochs": [100],
        "-batch_size": [128],
        "-epoch_to_start_align": [11],
        "-initialize_model": [1],
        "-init_batch_size": [32],
        "-init_lr": [10 ** -2.5],
        "-refinement": [refinement],
        "-n_epochs_refinement": [n_epochs_refinement],
        "-lambda_regul": lambda_regul,
        "-threshold_value": threshold_value,
        "-random_seed": random_seed
    }


class DannImputMNISTMNISTM(object):
    MAX_NB_PROCESSES = 2
    DEBUG = False
    BINARY = "experiments/launcher/digits_binary.py"
    GRID = {
        "-mode": ["dann_imput"],
        "-adaptive_lr": [1],
        "-source": ["MNIST"],
        "-target": ["MNISTM"],
        "-epoch_to_start_align": [11],
        "-is_balanced": [1],
        "-stop_grad": [0],
        "-n_epochs": [100],
        "-batch_size": [128],
        "-initialize_model": [1],
        "-init_batch_size": [32],
        "-bigger_reconstructor": [1],
        "-weight_d2": [weight_d2],
        "-weight_mse": [weight_mse],
        "-activate_mse": [activate_mse],
        "-activate_adaptation_imp": [activate_adaptation_imp],
        "-activate_adaptation_d1": [activate_adaptation_d1],
        "-bigger_discrim": [0],
        "-init_lr": [10 ** -2],
        "-refinement": [refinement],
        "-n_epochs_refinement": [n_epochs_refinement],
        "-lambda_regul": lambda_regul,
        "-lambda_regul_s": lambda_regul_s,
        "-threshold_value": threshold_value,
        "-random_seed": random_seed
    }


class DjdotMNISTMNISTM(object):
    MAX_NB_PROCESSES = 2
    DEBUG = False
    BINARY = "experiments/launcher/digits_binary.py"
    GRID = {
        "-mode": ["djdot"],
        "-upper_bound": [1],
        "-adaptive_lr": [1],
        "-is_balanced": [1],
        "-djdot_alpha": [0.1],
        "-source": ["MNIST"],
        "-target": ["MNISTM"],
        "-n_epochs": [100],
        "-batch_size": [500],
        "-initialize_model": [1],
        "-init_batch_size": [32],
        "-epoch_to_start_align": [11],
        "-init_lr": [10 ** -2],
        "-random_seed": random_seed
    }


class DjdotIgnoreMNISTMNISTM(object):
    MAX_NB_PROCESSES = 2
    DEBUG = False
    BINARY = "experiments/launcher/digits_binary.py"
    GRID = {
        "-mode": ["djdot"],
        "-upper_bound": [1],
        "-adaptive_lr": [1],
        "-is_balanced": [1],
        "-djdot_alpha": [0.1],
        "-source": ["MNIST"],
        "-target": ["MNISTM"],
        "-n_epochs": [100],
        "-batch_size": [500],
        "-initialize_model": [1],
        "-init_batch_size": [32],
        "-epoch_to_start_align": [11],
        "-init_lr": [10 ** -2],
        "-adapt_only_first": [1],
        "-random_seed": random_seed
    }


class DjdotZeroImputMNISTMNISTM(object):
    MAX_NB_PROCESSES = 2
    DEBUG = False
    BINARY = "experiments/launcher/digits_binary.py"
    GRID = {
        "-mode": ["djdot"],
        "-upper_bound": [0],
        "-adaptive_lr": [1],
        "-is_balanced": [1],
        "-djdot_alpha": [0.1],
        "-source": ["MNIST"],
        "-target": ["MNISTM"],
        "-n_epochs": [100],
        "-batch_size": [500],
        "-initialize_model": [1],
        "-init_batch_size": [32],
        "-epoch_to_start_align": [11],
        "-init_lr": [10 ** -2],
        "-random_seed": random_seed
    }


class DjdotImputMNISTMNISTM(object):
    MAX_NB_PROCESSES = 2
    DEBUG = False
    BINARY = "experiments/launcher/digits_binary.py"
    GRID = {
        "-mode": ["djdot_imput"],
        "-adaptive_lr": [1],
        "-source": ["MNIST"],
        "-target": ["MNISTM"],
        "-is_balanced": [1],
        "-epoch_to_start_align": [11],
        "-stop_grad": [1],
        "-djdot_alpha": [0.1],
        "-n_epochs": [100],
        "-batch_size": [500],
        "-initialize_model": [1],
        "-bigger_reconstructor": [1],
        "-init_batch_size": [32],
        "-init_lr": [10 ** -2],
        "-activate_mse": [activate_mse],
        "-activate_adaptation_imp": [activate_adaptation_imp],
        "-activate_adaptation_d1": [activate_adaptation_d1],
        "-random_seed": random_seed
    }
