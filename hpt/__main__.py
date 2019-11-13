import click

from hpt import TuningPlan
from deephub.common.io import load_yaml


@click.group()
def cli():
    pass


@cli.command()
@click.argument("hpt_fname", type=click.Path(exists=True, file_okay=True, dir_okay=True, readable=True))
@click.option('-p', '--param-expr', 'extra_expr_params', multiple=True, nargs=2,
              help="Override product configuration by setting a single value. "
                   "The format is expected to be to -p <KEY> <VALUE>. The key can be any dot separated path "
                   "for a product key e.g. 'train.epochs'. The value can be any string or even python expression "
                   "e.g. '100' or '100/5' or ['one', 'two']")
def run(hpt_fname, extra_expr_params):
    """
    Perform hyper parameter tuning based on a provided YAML file.

    hpt run <filename.yaml>


    Example of YAML file:

    \b
    ---------------------------------------
    goal: MAXIMIZE
    metric: "loss"
    max_trials: 30
    objective:
      type: MODULE_CALL
      module: deviceid.variants.triplet_loss_n_cluster
      function: train
    params:
      epochs:
        type: INTEGER
        constant: 1
      device_min_support:
        type: INTEGER
        min: 2
        max: 4
      embeddings_l2:
        type: REAL
        min: 0.00001
        max: 0.001
        scale: LOG

    """
    yaml_variables = load_yaml(hpt_fname)
    extra_params = {}
    extra_params['model'] = yaml_variables.get('fixed_model_params')
    extra_params['train'] = yaml_variables.get('fixed_train_params')

    for param, val in extra_expr_params:
        param_splitted = param.split(".")
        if param_splitted[0] == "fixed_model_params":
            extra_params['model']['.'.join([str(x) for x in param_splitted[1:]])] = val
        if param_splitted[0] == "fixed_train_params":
            extra_params['train']['.'.join([str(x) for x in param_splitted[1:]])] = val
    plan = TuningPlan.load_from_file(hpt_fname)
    best = plan.optimize(**extra_params)
    print(best)


if __name__ == '__main__':
    cli()
    # run()
