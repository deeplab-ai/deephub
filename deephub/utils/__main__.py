import click
import time

from deephub.common.io import resolve_glob_pattern
from deephub.models.feeders.tfrecords.meta import generate_fileinfo, get_fileinfo, TFRecordValidationError, \
    TFRecordInfoMissingError


@click.group()
def cli():
    """
    General purpose CLI utils.
    """
    pass


@cli.command()
@click.argument('pattern', type=str)
@click.option('--force', is_flag=True, default=False,
              help='It will forcefully regenerate meta data even for tfrecords that'
                   'have not changed.')
def generate_metadata(pattern, force):
    """
    Generate metadata for tfrecord files.

    With this util you can generate metadata from tfrecords based on a matching
    glob pattern.

    Example: Generate metadata for training dataset
      deep utils generate-metadata 'dataset/train-*'
    """

    files = resolve_glob_pattern(pattern)
    click.echo(f"{len(files)} files matched with the pattern.")

    with click.progressbar(files) as files:
        for fpath in files:
            try:
                generate_fileinfo(fpath)
            except Exception as e:
                click.echo(f'Skipping file {fpath} because of: {e!s}')
    click.echo('Finished generating metadata')


@cli.command()
@click.argument('pattern', type=str)
def total_examples(pattern) -> int:
    """
    Get total examples for all the files matched with the given input file pattern.
    """

    files = resolve_glob_pattern(pattern)
    click.echo(f"{len(files)} files matched with the pattern.")

    total_rows = 0
    for file in files:
        try:
            total_rows += get_fileinfo(file).total_records
        except Exception:
            pass

    click.echo(f"Total number of examples: {total_rows}")


@cli.command()
@click.argument('pattern', type=str)
@click.option('--shallow-check/--deep-check', default=True,
              help='Flag in order to control shallow or deep md5 hash check. With shallow-check only the size of'
                   'each file will be validated, while with deep-check both size and md5 hash of each file will be'
                   'validated.')
def validate(pattern: str, shallow_check: bool):
    """
    Validate each one of the files matched using the input file pattern.
    """
    start = time.time()

    files = resolve_glob_pattern(pattern)
    click.echo(f"{len(files)} files matched with the pattern.")

    with click.progressbar(files) as files:
        for file in files:
            try:
                get_fileinfo(file, shallow_check)  # inside here happens the validation step too
            except TFRecordValidationError:
                raise
            except TFRecordInfoMissingError:
                raise
            except Exception as e:  # Probably not a valid tfrecords file
                click.echo(f'Probably not a valid tf_record file {e}')

    end = time.time()

    click.echo(f"Total execution time: {end - start}")
