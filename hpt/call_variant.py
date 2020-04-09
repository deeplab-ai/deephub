import click
from deephub.trainer import Trainer
from deephub.variants.io import (load_variant, UnknownVariant)
from deephub.trainer.metrics import get_metrics_from_tensorboard
import os


def train_and_get_score(**params):

    if params.get('extra_str_params'):
        raise click.ClickException("--param-str is not yet implemented, use --param-expr instead.")

    # Load variant definition
    try:
        variant_definition = load_variant(variant_name=params.get('model').get('variant_name'),
                                          variants_dir=params.get('model').get('variant_path'))

    except UnknownVariant:
        raise click.ClickException(f"Cannot find variant with name '{params.get('variant_name')}'")

    # Override definition with user configuration
    for param in params.get('model'):
        v = params.get('model')[param]
        variant_definition.set('model.'+param, v)

    for param in params.get('train'):
        v = params.get('train')[param]
        variant_definition.set('train.'+param, v)

    if params.get('warm_start_checkpoint'):
        variant_definition.set('train.warm_start_check_point', params.get('warm_start_checkpoint'))
        variant_definition.set('train.warm_start_variables_regex', params.get('warm_start_vars'))
    params_str = "hpt/"
    if not os.path.exists(variant_definition.definition['model']['model_dir']):
        os.mkdir(variant_definition.definition['model']['model_dir'])
    if not os.path.exists(variant_definition.definition['model']['model_dir']+params_str):
        os.mkdir(variant_definition.definition['model']['model_dir']+params_str)
    for param in params:
        if param not in ["model", "train"]:
            params_str += "{}-{}.".format(param, params[param])
    variant_definition.definition['model']['model_dir'] += params_str
    trainer = Trainer(requested_gpu_devices=params.get('gpu_device_ids'))

    variant_definition.train(trainer=trainer)
    metrics_dict = get_metrics_from_tensorboard(variant_definition.definition['model']['model_dir'])
    return metrics_dict



