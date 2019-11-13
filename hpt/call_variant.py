import click

from deephub.trainer import Trainer
from deephub.variants.io import (load_variant, UnknownVariant)


import tensorflow as tf
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

    trainer = Trainer(requested_gpu_devices=params.get('gpu_device_ids'))
    variant_definition.train(trainer=trainer)
    metrics_dict = get_metrics_from_tensorboard(variant_definition.definition['model']['model_dir'])

    return metrics_dict


def get_metrics_from_tensorboard(model_dir):
    # Parse events out tensorboard files
    events_out_files = [os.path.join(model_dir, 'eval', x) for x in os.listdir(os.path.join(model_dir, 'eval'))]
    metrics_dict = {}
    for events_out_file in events_out_files:
        for e in tf.train.summary_iterator(events_out_file):
            for v in e.summary.value:
                try:
                    metrics_dict[v.tag].append(float('{:.7f}'.format(v.simple_value)))
                except KeyError:
                    metrics_dict[v.tag] = [float('{:.7f}'.format(v.simple_value))]
    return metrics_dict




