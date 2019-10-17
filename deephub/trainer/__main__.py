import click

from deephub.trainer import Trainer
from deephub.variants.io import (load_variant, UnknownVariant)


@click.group()
def cli():
    pass


@cli.command()
@click.argument('variant_name', type=str)
@click.option('-d', '--variants-dir', 'variants_dir', type=str, default=None,
              help="The directory where are the YAML files with variant definition.")
@click.option("-g", "--gpu-device-id", "gpu_device_ids", type=int, default=None, multiple=True,
              help="The index of the GPU device id to use for training (in CUDA ORDER). Can be used "
                   "multiple times to select multiple GPUs")
@click.option('-p', '--param-expr', 'extra_expr_params', multiple=True, nargs=2,
              help="Override variant configuration by setting a single value. "
                   "The format is expected to be to -p <KEY> <VALUE>. The key can be any dot separated path "
                   "for a variant key e.g. 'train.epochs'. The value can be any string or even python expression "
                   "e.g. '100' or '100/5' or ['one', 'two']")
@click.option('-P', '--param-str', 'extra_str_params', multiple=True, nargs=2,
              help="Override variant configuration by setting a single value. "
                   "The format is the same as --param-expr but it does not evaluate the value, it is used as is.")
@click.option('--warm-start-checkpoint', type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True),
              help="The directory with checkpoint file(s) or path to checkpoint "
                   "from which to warm-start the model parameters.")
@click.option('--warm-start-vars', type=str, default='.*',
              help="A regular expression (string) that captures which variables to warm-start (see tf.get_collection)."
                   "This expression will only consider variables in the TRAINABLE_VARIABLES collection. "
                   "Must be used with --warm-start-checkpoint. "
                   "Eg. if we feed 'InceptionV3/Logits|InceptionV3/AuxLogits|InceptionV3/Mixed_7c'"
                   " we will restore only these 3 scopes from inception_v3.ckpt. "
                   "If we feed '^(?!InceptionV3/Logits|InceptionV3/AuxLogits|InceptionV3/Mixed_7c)' "
                   "we will restore all the scopes except these 3.")
def train(variant_name, variants_dir, gpu_device_ids, extra_expr_params, extra_str_params,
          warm_start_checkpoint, warm_start_vars):
    """
    Train a variant from packaged definitions or from a user provided.

    VARIANT_NAME: The unique name of the variant as it was defined in the YAML definitions file.

    \b
    Features:
    ========

    \b
    Warm-Starting:
    -------------

    You can optionally warm-start a model before training using the
    variable values of another model. Weights are usually loaded
    from  a specific checkpoint can be provided (in the case of
    the former, the latest checkpoint will be used).

    You can control which variables to warm-start and which to skip
    using a regular expression to match their names.

    Example to warm-start ALL variables:

      $ deep trainer train some_variant --warm-start-checkpoint /path/to/checkpoint
    --warm-start-vars '.*'

    Example to warm-start only the embeddings (input layer):

      $ deep trainer train some_variant --warm-start-checkpoit /path/to/checkpoint
    --warm-start-vars '.*input_layer.*'

    Example to warm-start all layers but the

    \b
    GPU Support:
    ------------

    By default trainer will be performed on CPU unless the user has
    explicitly requested.

    The current trainer has full support for GPU training as long as the
    correct tensorflow library is installed and a CUDA device is accessible.

    To train on single GPU use one the `-g` option. E.g.

      $ deep trainer train some_variant -g 0

    To train on multi GPU, it is as easy as using multiple times the -g option

      $ deep trainer train some_variant -g 0 -g 2

    *Notice* The device id must be in the CUDA order and not that of nvidia-smi.
    """

    if extra_str_params:
        raise click.ClickException("--param-str is not yet implemented, use --param-expr instead.")

    # Load variant definition
    try:
        variant_definition = load_variant(variant_name=variant_name, variants_dir=variants_dir)
    except UnknownVariant:
        raise click.ClickException(f"Cannot find variant with name '{variant_name}'")

    # Override definition with user configuration
    for param in extra_expr_params:
        k, v = param
        # Try to parse expression on right part, else it is considered a raw string
        try:
            v = eval(v)
        except Exception:
            pass
        variant_definition.set(k, v)

    if warm_start_checkpoint:
        variant_definition.set('train.warm_start_check_point', warm_start_checkpoint)
        variant_definition.set('train.warm_start_variables_regex', warm_start_vars)

    trainer = Trainer(requested_gpu_devices=gpu_device_ids)
    click.echo("Starting model training.")

    variant_definition.train(trainer=trainer)
    click.echo('Training has finished.')


if __name__ == '__main__':
    cli()
