# flake8: noqa E501
import logging
import os
import sys

import click
import tensorflow as tf

from deephub.common.logging import ClickStreamHandler, tf_cpp_log_level
from deephub.resources.__main__ import cli as resources_cli
from deephub.utils.__main__ import cli as utils_cli
from deephub.trainer.__main__ import cli as tf_trainer_cli

# I know this is ugly here, but its beautiful on the terminal (kpal)
banner_txt = f"""
       &&(       
        &(     
   &&{click.style(fg="blue", text=r"   .  ,")}&(      ___                        {click.style(fg="blue", text=r" _           _     ")}
 &%{click.style(fg="blue", text=r"   *   ,   ")}&(   (  _`\                      {click.style(fg="blue", text=r"(_ )        ( )    ")}
&/{click.style(fg="blue", text=r"  *  .,  ,.  ")}&%  | | ) |   __     __   _ _   {click.style(fg="blue", text=r" | |    _ _ | |_   ")}
{click.style(fg="blue", text=r" /")}&.{click.style(fg="blue", text=r"  *   *  ")}%&    | | | ) /'__`\ /'__`\( '_`\ {click.style(fg="blue", text=r" | |  /'_` )| '_`Δ ")}
{click.style(fg="blue", text=r"// /")}&*{click.style(fg="blue", text=r" .,  ")}&&  *   | |_) |(  ___/(  ___/| (_) ){click.style(fg="blue", text=r" | | ( (_| || |_) )")}
{click.style(fg="blue", text=r"  (, /")}&. %&{click.style(fg="blue", text=r"  ,")}     (____/'`\____)`\____)| ,__/'{click.style(fg="blue", text=r"(___)`Δ__,_)(_,__/'")}
{click.style(fg="blue", text=r"    // ")}(%{click.style(fg="blue", text=r"  (")}                            | |    {click.style(fg="blue", text=r"                   ")}
{click.style(fg="blue", text=r"      (, (,")}                             (_)                       
""".replace('Δ', '\\')  # This is a hack because: f-string expression part cannot include a backslash


@click.group()
@click.option('--logging-level', '-l',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']),
              default='WARNING', help='Define the logging level')
@click.option('--logging-stream',
              type=click.Choice(['stdout', 'stderr']),
              default='stderr', help='Define the output stream to use for printing log events')
@click.option('--banner/--hide-banner', default=True,
              help="Hide the main banner of the project")
def cli(logging_level, logging_stream, banner):
    if banner:
        click.echo(banner_txt)

    log_level_no = getattr(logging, logging_level)

    logging.getLogger().setLevel(log_level_no)

    # Configure tensorflow core
    tf.logging.set_verbosity(log_level_no)
    logging.getLogger('tensorflow').handlers = []

    # Configure tensorflow cpp logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf_cpp_log_level(log_level_no))

    logging_handler = ClickStreamHandler(stream=getattr(sys, logging_stream))
    logging_formatter = logging.Formatter("%(asctime)s - [%(levelname)-07s] %(message)s - %(name)s ")
    logging_handler.setFormatter(logging_formatter)

    logging.getLogger().addHandler(logging_handler)


cli.add_command(resources_cli, name="resources")
cli.add_command(tf_trainer_cli, name='trainer')
# cli.add_command(serving_cli, name='serving')
cli.add_command(utils_cli, name='utils')

if __name__ == '__main__':
    cli()
