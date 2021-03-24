import typing
import json
import argparse


class RunConfig(typing.NamedTuple):
    experiment_name: str = "test"
    metarun_id: int = 0
    run_id: int = 0
    max_nb_processes: int = 2
    gpu_id: int = 0


class ModelConfig(typing.NamedTuple):
    mode: str
    upper_bound: int = 1
    adaptive_lr: int = 0
    djdot_alpha: float = 1.0
    source: str = "SVHN"
    target: str = "MNIST"
    epoch_to_start_align: int = 11
    stop_grad: int = 0
    bigger_reconstructor: int = 0
    bigger_discrim: int = 1
    is_balanced: int = 0
    initialize_model: int = 0
    random_seed: int = 1985
    init_lr: float = 0.002
    output_fig: int = 0
    use_categorical: int = 1
    crop_ratio: float = 0.5
    adapt_only_first: int = 0
    adaptive_grad_scale: int = 1
    smaller_lr_imput: int = 0
    weight_d2: float = 1.0
    weight_mse: float = 1.0
    weight_classif: float = 1.0
    activate_adaptation_d1: int = 1
    activate_mse: int = 1
    activate_adaptation_imp: int = 1
    refinement: int = 0
    n_epochs_refinement: int = 10
    lambda_regul: float = 1.0
    lambda_regul_s: float = 1.0
    threshold_value: float = 0.95


class TrainingConfig(typing.NamedTuple):
    n_epochs: int = 20
    batch_size: int = 128
    test_batch_size: int = 1000
    init_batch_size: int = 128


class DatasetConfig(typing.NamedTuple):
    channel: int = 1
    im_size: int = 28


class Config(typing.NamedTuple):
    run: RunConfig
    model: ModelConfig
    training: TrainingConfig

    def json_encode(self):
        return json.dumps({
            "run": self.run._asdict(),
            "model": self.model._asdict(),
            "training": self.training._asdict()
        })

    @staticmethod
    def json_decode(json_string: str):
        data = json.loads(json_string)

        return Config(
            run=RunConfig(**data["run"]),
            model=ModelConfig(**data["model"]),
            training=TrainingConfig(**data["training"])
        )

    @staticmethod
    def get_config_from_args() -> "Config":
        parser = argparse.ArgumentParser(description='Learn classifier for domain adaptation')

        parser.add_argument('--experiment_name', type=str)
        parser.add_argument('--metarun_id', type=int)
        parser.add_argument('--run_id', type=int)
        parser.add_argument('--max_nb_processes', type=int)
        parser.add_argument('--gpu_id', type=int, default=0)
        parser.add_argument('--mode', type=str)
        parser.add_argument('--upper_bound', type=int, default=1)
        parser.add_argument('--djdot_alpha', type=float)
        parser.add_argument('--adaptive_lr', type=int)
        parser.add_argument('--adaptive_grad_scale', type=int, default=1)
        parser.add_argument('--epoch_to_start_align', type=int, default=11)
        parser.add_argument('--stop_grad', type=int)
        parser.add_argument('--source', type=str)
        parser.add_argument('--target', type=str)
        parser.add_argument('--n_epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument('--bigger_reconstructor', type=int, default=0)
        parser.add_argument('--bigger_discrim', type=int, default=1)
        parser.add_argument('--is_balanced', type=int, default=0)
        parser.add_argument('--test_batch_size', type=int, default=1000)
        parser.add_argument('--initialize_model', type=int, default=0)
        parser.add_argument('--init_batch_size', type=int, default=128)
        parser.add_argument('--random_seed', type=int, default=1985)
        parser.add_argument('--init_lr', type=float, default=0.002)
        parser.add_argument('--output_fig', type=int, default=0)
        parser.add_argument('--use_categorical', type=int, default=1)
        parser.add_argument('--crop_ratio', type=float, default=0.5)
        parser.add_argument('--adapt_only_first', type=int, default=0)
        parser.add_argument('--smaller_lr_imput', type=int, default=0)
        parser.add_argument('--weight_mse', type=float, default=1.0)
        parser.add_argument('--weight_d2', type=float, default=1.0)
        parser.add_argument('--activate_adaptation_imp', type=int, default=1)
        parser.add_argument('--activate_adaptation_d1', type=int, default=1)
        parser.add_argument('--activate_mse', type=int, default=1)
        parser.add_argument('--weight_classif', type=float, default=1.0)
        parser.add_argument('--refinement', type=int, default=0)
        parser.add_argument('--n_epochs_refinement', type=int, default=10)
        parser.add_argument('--lambda_regul', type=float, default=1.0)
        parser.add_argument('--threshold_value', type=float, default=0.95)
        parser.add_argument('--lambda_regul_s', type=float, default=1.0)

        args, _ = parser.parse_known_args()

        run_config = RunConfig(
            experiment_name=args.experiment_name,
            metarun_id=args.metarun_id,
            run_id=args.run_id,
            max_nb_processes=args.max_nb_processes,
            gpu_id=args.gpu_id
        )

        model_config = ModelConfig(
            mode=args.mode,
            upper_bound=args.upper_bound,
            djdot_alpha=args.djdot_alpha,
            adaptive_lr=args.adaptive_lr,
            epoch_to_start_align=args.epoch_to_start_align,
            source=args.source,
            target=args.target,
            stop_grad=args.stop_grad,
            bigger_reconstructor=args.bigger_reconstructor,
            bigger_discrim=args.bigger_discrim,
            is_balanced=args.is_balanced,
            initialize_model=args.initialize_model,
            random_seed=args.random_seed,
            init_lr=args.init_lr,
            output_fig=args.output_fig,
            use_categorical=args.use_categorical,
            crop_ratio=args.crop_ratio,
            adapt_only_first=args.adapt_only_first,
            adaptive_grad_scale=args.adaptive_grad_scale,
            smaller_lr_imput=args.smaller_lr_imput,
            activate_adaptation_imp=args.activate_adaptation_imp,
            activate_adaptation_d1=args.activate_adaptation_d1,
            activate_mse=args.activate_mse,
            weight_mse=args.weight_mse,
            weight_d2=args.weight_d2,
            weight_classif=args.weight_classif,
            refinement=args.refinement,
            n_epochs_refinement=args.n_epochs_refinement,
            lambda_regul=args.lambda_regul,
            threshold_value=args.threshold_value,
            lambda_regul_s=args.lambda_regul_s
        )

        training_config = TrainingConfig(
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            init_batch_size=args.init_batch_size
        )

        my_config = Config(
            run=run_config,
            model=model_config,
            training=training_config
        )

        return my_config


dummy_model_config = Config(
    run=RunConfig(gpu_id=0, run_id=0, metarun_id=5),
    model=ModelConfig(mode="dann", upper_bound=1, source="USPS", target="MNIST", epoch_to_start_align=0,
                      is_balanced=0, stop_grad=1, djdot_alpha=0.1, initialize_model=0, refinement=1,
                      n_epochs_refinement=10),
    training=TrainingConfig(n_epochs=0, batch_size=500, test_batch_size=500, init_batch_size=32)
)
