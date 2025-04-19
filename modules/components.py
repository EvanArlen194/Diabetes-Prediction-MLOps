import os
import tensorflow_model_analysis as tfma

from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Evaluator,
    Pusher
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy
)


def create_split_config(train_ratio: int = 8, eval_ratio: int = 2) -> example_gen_pb2.SplitConfig:
    """
    Creates a split configuration for the dataset.

    Args:
        train_ratio (int): Proportion of data for training.
        eval_ratio (int): Proportion of data for evaluation.

    Returns:
        example_gen_pb2.SplitConfig: Configuration for splitting the data.
    """
    return example_gen_pb2.SplitConfig(splits=[
        example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=train_ratio),
        example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=eval_ratio),
    ])


def create_eval_config(label_key: str, slicing_features: list, metric_threshold: float) -> tfma.EvalConfig:
    """
    Creates an evaluation configuration for the model.

    Args:
        label_key (str): The label key in the dataset.
        slicing_features (list): Features to create slicing specs for evaluation.
        metric_threshold (float): Threshold for the `BinaryAccuracy` metric.

    Returns:
        tfma.EvalConfig: Evaluation configuration.
    """
    slicing_specs = [tfma.SlicingSpec()] + [tfma.SlicingSpec(feature_keys=[key]) for key in slicing_features]

    metrics_specs = tfma.MetricsSpec(metrics=[
        tfma.MetricConfig(class_name="AUC"),
        tfma.MetricConfig(class_name="Precision"),
        tfma.MetricConfig(class_name="Recall"),
        tfma.MetricConfig(class_name="ExampleCount"),
        tfma.MetricConfig(
            class_name="BinaryAccuracy",
            threshold=tfma.MetricThreshold(
                value_threshold=tfma.GenericValueThreshold(lower_bound={"value": metric_threshold}),
                change_threshold=tfma.GenericChangeThreshold(
                    direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                    absolute={"value": 0.0001}
                )
            )
        ),
    ])

    return tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key=label_key)],
        slicing_specs=slicing_specs,
        metrics_specs=[metrics_specs]
    )


def create_pipeline_components(
    data_dir: str,
    transform_module: str,
    trainer_module: str,
    train_steps: int,
    eval_steps: int,
    serving_model_dir: str,
    label_key: str = "diabetes",
    slicing_features: list = ["gender", "heart_disease"],
    metric_threshold: float = 0.5
):
    """
    Creates and initializes TFX pipeline components for model training and deployment.

    Args:
        data_dir (str): Path to the input data directory.
        transform_module (str): Path to the transformation module file.
        trainer_module (str): Path to the trainer module file.
        train_steps (int): Number of training steps.
        eval_steps (int): Number of evaluation steps.
        serving_model_dir (str): Directory to store the exported model for serving.
        label_key (str): The label key in the dataset.
        slicing_features (list): Features to create slicing specs for evaluation.
        metric_threshold (float): Threshold for the `BinaryAccuracy` metric.

    Returns:
        list: List of initialized TFX components.
    """
    # Create configurations
    split_config = create_split_config()
    eval_config = create_eval_config(label_key, slicing_features, metric_threshold)

    # Initialize TFX components
    example_gen = CsvExampleGen(
        input_base=data_dir,
        output_config=example_gen_pb2.Output(split_config=split_config)
    )

    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

    schema_gen = SchemaGen(statistics=statistics_gen.outputs["statistics"])

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"]
    )

    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=os.path.abspath(transform_module)
    )

    trainer = Trainer(
        module_file=os.path.abspath(trainer_module),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=train_steps),
        eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=eval_steps)
    )

    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id("latest_blessed_model_resolver")

    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config
    )

    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(base_directory=serving_model_dir)
        )
    )

    return [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher
    ]
