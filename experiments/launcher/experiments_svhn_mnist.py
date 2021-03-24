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


class DannSVHNMNIST(object):
    MAX_NB_PROCESSES = 2
    DEBUG = False
    BINARY = "experiments/launcher/digits_binary.py"
    GRID = {
        "-mode": ["dann"],
        "-upper_bound": [1],
        "-adaptive_lr": [1],
        "-is_balanced": [1],
        "-source": ["SVHN"],
        "-target": ["MNIST"],
        "-epoch_to_start_align": [11],
        "-n_epochs": [100],
        "-batch_size": [128],
        "-initialize_model": [0],
        "-init_batch_size": [32],
        "-init_lr": [10 ** -2],
        "-refinement": [refinement],
        "-n_epochs_refinement": [n_epochs_refinement],
        "-lambda_regul": lambda_regul,
        "-lambda_regul_s": lambda_regul_s,
        "-crop_ratio": [0.5],
        "-threshold_value": threshold_value,
        "-random_seed": random_seed
    }


class DannIgnoreSVHNMNIST(object):
    MAX_NB_PROCESSES = 2
    DEBUG = False
    BINARY = "experiments/launcher/digits_binary.py"
    GRID = {
        "-mode": ["dann"],
        "-upper_bound": [1],
        "-adaptive_lr": [1],
        "-is_balanced": [1],
        "-source": ["SVHN"],
        "-target": ["MNIST"],
        "-epoch_to_start_align": [11],
        "-n_epochs": [100],
        "-batch_size": [128],
        "-initialize_model": [0],
        "-init_batch_size": [32],
        "-init_lr": [10 ** -2],
        "-adapt_only_first": [1],
        "-crop_ratio": [0.5],
        "-refinement": [refinement],
        "-n_epochs_refinement": [n_epochs_refinement],
        "-lambda_regul": lambda_regul,
        "-threshold_value": threshold_value,
        "-random_seed": random_seed
    }


class DannZeroImputSVHNMNIST(object):
    MAX_NB_PROCESSES = 2
    DEBUG = False
    BINARY = "experiments/launcher/digits_binary.py"
    GRID = {
        "-mode": ["dann"],
        "-upper_bound": [0],
        "-adaptive_lr": [1],
        "-is_balanced": [1],
        "-source": ["SVHN"],
        "-target": ["MNIST"],
        "-epoch_to_start_align": [11],
        "-n_epochs": [100],
        "-batch_size": [128],
        "-initialize_model": [0],
        "-init_batch_size": [32],
        "-init_lr": [10 ** -2.5],
        "-crop_ratio": [0.5],
        "-refinement": [refinement],
        "-n_epochs_refinement": [n_epochs_refinement],
        "-lambda_regul": lambda_regul,
        "-threshold_value": threshold_value,
        "-random_seed": random_seed
    }


class DannImputSVHNMNIST(object):
    MAX_NB_PROCESSES = 2
    DEBUG = False
    BINARY = "experiments/launcher/digits_binary.py"
    GRID = {
        "-mode": ["dann_imput"],
        "-adaptive_lr": [1],
        "-source": ["SVHN"],
        "-target": ["MNIST"],
        "-epoch_to_start_align": [11],
        "-is_balanced": [1],
        "-stop_grad": [0],
        "-n_epochs": [100],
        "-batch_size": [128],
        "-initialize_model": [0],
        "-init_batch_size": [32],
        "-bigger_reconstructor": [1],
        "-weight_d2": [weight_d2],
        "-weight_mse": [weight_mse],
        "-activate_mse": [activate_mse],
        "-activate_adaptation_imp": [activate_adaptation_imp],
        "-activate_adaptation_d1": [activate_adaptation_d1],
        "-bigger_discrim": [0],
        "-init_lr": [10 ** -2],
        "-crop_ratio": [0.5],
        "-refinement": [refinement],
        "-n_epochs_refinement": [n_epochs_refinement],
        "-lambda_regul": lambda_regul,
        "-lambda_regul_s": lambda_regul_s,
        "-threshold_value": threshold_value,
        "-random_seed": random_seed
    }


class DjdotSVHNMNIST(object):
    MAX_NB_PROCESSES = 2
    DEBUG = False
    BINARY = "experiments/launcher/digits_binary.py"
    GRID = {
        "-mode": ["djdot"],
        "-upper_bound": [1],
        "-adaptive_lr": [1],
        "-is_balanced": [1],
        "-djdot_alpha": [0.1],
        "-source": ["SVHN"],
        "-target": ["MNIST"],
        "-epoch_to_start_align": [11],
        "-n_epochs": [50],
        "-batch_size": [500],
        "-initialize_model": [0],
        "-init_batch_size": [32],
        "-init_lr": [10 ** -2],
        "-crop_ratio": [0.5],
        "-random_seed": random_seed
    }


class DjdotIgnoreSVHNMNIST(object):
    MAX_NB_PROCESSES = 2
    DEBUG = False
    BINARY = "experiments/launcher/digits_binary.py"
    GRID = {
        "-mode": ["djdot"],
        "-upper_bound": [1],
        "-adaptive_lr": [1],
        "-is_balanced": [1],
        "-djdot_alpha": [0.1],
        "-source": ["SVHN"],
        "-target": ["MNIST"],
        "-epoch_to_start_align": [11],
        "-n_epochs": [50],
        "-batch_size": [500],
        "-initialize_model": [0],
        "-init_batch_size": [32],
        "-init_lr": [10 ** -2],
        "-adapt_only_first": [1],
        "-crop_ratio": [0.5],
        "-random_seed": random_seed
    }


class DjdotZeroImputSVHNMNIST(object):
    MAX_NB_PROCESSES = 2
    DEBUG = False
    BINARY = "experiments/launcher/digits_binary.py"
    GRID = {
        "-mode": ["djdot"],
        "-upper_bound": [0],
        "-adaptive_lr": [1],
        "-is_balanced": [1],
        "-djdot_alpha": [0.1],
        "-source": ["SVHN"],
        "-target": ["MNIST"],
        "-epoch_to_start_align": [11],
        "-n_epochs": [50],
        "-batch_size": [500],
        "-initialize_model": [0],
        "-init_batch_size": [32],
        "-init_lr": [10 ** -2],
        "-crop_ratio": [0.5],
        "-random_seed": random_seed
    }


class DjdotImputSVHNMNIST(object):
    MAX_NB_PROCESSES = 2
    DEBUG = False
    BINARY = "experiments/launcher/digits_binary.py"
    GRID = {
        "-mode": ["djdot_imput"],
        "-adaptive_lr": [1],
        "-source": ["SVHN"],
        "-target": ["MNIST"],
        "-epoch_to_start_align": [11],
        "-is_balanced": [1],
        "-stop_grad": [1],
        "-djdot_alpha": [0.1],
        "-n_epochs": [50],
        "-batch_size": [500],
        "-initialize_model": [0],
        "-bigger_reconstructor": [1],
        "-activate_mse": [activate_mse],
        "-activate_adaptation_imp": [activate_adaptation_imp],
        "-activate_adaptation_d1": [activate_adaptation_d1],
        "-init_batch_size": [32],
        "-init_lr": [10 ** -2],
        "-crop_ratio": [0.5],
        "-random_seed": random_seed
    }
