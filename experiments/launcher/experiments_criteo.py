use_categorical = 0
n_epochs = 50

activate_adaptation_imp = 1
activate_mse = 1
activate_adaptation_d1 = 1
weight_d2 = 1.0
weight_mse = 0.005
refinement = 1
n_epochs_refinement = 10
lambda_regul = [1.0]
lambda_regul_s = [1.0]
is_balanced = 0
threshold_value = [0.95]

compute_variance = False
random_seed = [1985] if not compute_variance else [1985, 2184, 51, 12, 465]


class SourceZeroImputCriteo(object):
    MAX_NB_PROCESSES = 1
    DEBUG = False
    BINARY = "experiments/launcher/criteo_binary.py"
    GRID = {
        "-mode": ["dann"],
        "-upper_bound": [0],
        "-n_epochs": [n_epochs],
        "-epoch_to_start_align": [n_epochs],
        "-adaptive_lr": [1],
        "-batch_size": [500],
        "-initialize_model": [1],
        "-init_batch_size": [500],
        "-init_lr": [10 ** -6],
        "-use_categorical": [use_categorical],
        "-random_seed": random_seed
    }


class SourceIgnoreCriteo(object):
    MAX_NB_PROCESSES = 1
    DEBUG = False
    BINARY = "experiments/launcher/criteo_binary.py"
    GRID = {
        "-mode": ["dann"],
        "-n_epochs": [n_epochs],
        "-epoch_to_start_align": [n_epochs],
        "-adaptive_lr": [1],
        "-batch_size": [500],
        "-initialize_model": [1],
        "-init_batch_size": [500],
        "-init_lr": [10 ** -6],
        "-use_categorical": [use_categorical],
        "-adapt_only_first": [1],
        "-random_seed": random_seed
    }


class DannZeroImputCriteo(object):
    MAX_NB_PROCESSES = 2
    DEBUG = False
    BINARY = "experiments/launcher/criteo_binary.py"
    GRID = {
        "-mode": ["dann"],
        "-upper_bound": [0],
        "-adaptive_lr": [1],
        "-n_epochs": [n_epochs],
        "-epoch_to_start_align": [6],
        "-batch_size": [500],
        "-is_balanced": [is_balanced],
        "-initialize_model": [1],
        "-init_batch_size": [500],
        "-init_lr": [10 ** -6],
        "-use_categorical": [use_categorical],
        "-bigger_discrim": [1],
        "-refinement": [refinement],
        "-n_epochs_refinement": [n_epochs_refinement],
        "-lambda_regul": lambda_regul,
        "-lambda_regul_s": lambda_regul_s,
        "-threshold_value": threshold_value,
        "-random_seed": random_seed
    }


class DannIgnoreCriteo(object):
    MAX_NB_PROCESSES = 2
    DEBUG = False
    BINARY = "experiments/launcher/criteo_binary.py"
    GRID = {
        "-mode": ["dann"],
        "-adaptive_lr": [1],
        "-n_epochs": [n_epochs],
        "-epoch_to_start_align": [6],
        "-batch_size": [500],
        "-initialize_model": [1],
        "-is_balanced": [is_balanced],
        "-init_batch_size": [500],
        "-init_lr": [10 ** -6],
        "-use_categorical": [use_categorical],
        "-bigger_discrim": [1],
        "-adapt_only_first": [1],
        "-refinement": [refinement],
        "-n_epochs_refinement": [n_epochs_refinement],
        "-lambda_regul": lambda_regul,
        "-lambda_regul_s": lambda_regul_s,
        "-threshold_value": threshold_value,
        "-random_seed": random_seed
    }


class DannImputCriteo(object):
    MAX_NB_PROCESSES = 2
    DEBUG = False
    BINARY = "experiments/launcher/criteo_binary.py"
    GRID = {
        "-mode": ["dann_imput"],
        "-upper_bound": [0],
        "-n_epochs": [n_epochs],
        "-epoch_to_start_align": [6],
        "-adaptive_lr": [1],
        "-adaptive_grad_scale": [1],
        "-stop_grad": [0],
        "-is_balanced": [is_balanced],
        "-batch_size": [500],
        "-init_lr": [10 ** -6],
        "-initialize_model": [1],
        "-init_batch_size": [500],
        "-bigger_reconstructor": [1],
        "-bigger_discrim": [1],
        "-use_categorical": [use_categorical],
        "-activate_adaptation_imp": [activate_adaptation_imp],
        "-activate_mse": [activate_mse],
        "-activate_adaptation_d1": [activate_adaptation_d1],
        "-smaller_lr_imput": [1],
        "-weight_d2": [weight_d2],
        "-weight_mse": [weight_mse],
        "-weight_classif": [1.0],
        "-refinement": [refinement],
        "-n_epochs_refinement": [n_epochs_refinement],
        "-lambda_regul": lambda_regul,
        "-lambda_regul_s": lambda_regul_s,
        "-threshold_value": threshold_value,
        "-random_seed": random_seed
    }
