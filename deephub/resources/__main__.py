from __future__ import absolute_import
from __future__ import print_function

import click
from tabulate import tabulate

from deephub.common.utils import size_formatted
from .remote import get_resource, get_all_resources


@click.group()
def cli():
    """
    Manage external resources that are need by deeper.
    """
    pass


@cli.command()
@click.argument(u'resource_id', type=click.Choice([a.name for a in get_all_resources()] + ['ALL']))
def download(resource_id):
    """
    Download locally resources from google storage.

    Some optional functionality of the code base depends on extra resources that are only available in external
    cloud storage. This tool helps download this functionality and store in a local directory.
    """

    try:
        if resource_id == 'ALL':
            for resource in get_all_resources():
                resource.download()
        else:
            get_resource(resource_id).download()
    except Exception as e:
        raise click.ClickException(e.message)


@cli.command()
def list():
    """
    List all resources that are downloadable from external resources.
    """

    table = [
        [resource.name,

         'Recursive' if resource.is_directory else 'File',
         size_formatted(resource.remote_size()),
         resource.gs_url, ]
        for resource in get_all_resources()
    ]

    print(tabulate(
        table,
        headers=['RESOURCE_ID', 'Type', 'Size', 'Source URL'],
    ))


if __name__ == '__main__':
    cli()
