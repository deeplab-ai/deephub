import csv
import logging
from functools import reduce
from itertools import groupby
from typing import Dict, Union, Optional, Tuple

import tensorflow as tf

from deephub.common import io

logger = logging.getLogger(__name__)


def export_metrics(model_dir: str) -> None:
    """
    Export training metrics to a csv file in the model directory
    :param model_dir: The model directory to save the csv file in
    """
    train_events_file_paths = io.resolve_glob_pattern("*events.out.tfevents*", model_dir)
    validation_events_file_paths = io.resolve_glob_pattern("eval/*events.out.tfevents*", model_dir)

    if len(train_events_file_paths) == 1:
        train_events_file_path = str(train_events_file_paths[0])
        validation_events_file_path = str(validation_events_file_paths[0]) if len(
            validation_events_file_paths) == 1 else None

        output_file_path = f"{model_dir}/metrics.csv"
        _export_last_step_event_metrics_to_csv(train_events_file_path=train_events_file_path,
                                               validation_events_file_path=validation_events_file_path,
                                               output_file_path=output_file_path)

        logger.info("Exported metrics in %s", output_file_path)
    else:
        logger.warning("Could not export metrics because a single train events file should be found in %s", model_dir)


# TODO unit tests
# TODO decouple merging of train and validation columns from csv creation
def _export_last_step_event_metrics_to_csv(train_events_file_path: str,
                                           validation_events_file_path: Optional[str], output_file_path: str,
                                           black_listed_tags: Optional[Tuple[str]] =
                                           ('global_step/sec', 'checkpoint_path')) -> None:
    """
    Reads the training and the validation tensorboard event files, parses the last step summary as metrics and exports
    them to a csv file
    :param train_events_file_path: path to the train events file
    :param validation_events_file_path: path to the validation events file
    :param output_file_path: output file name
    :param black_listed_tags: event summary tags to exclude
    """

    def last_step_summary_in_events(events_file: str, data_set_type: str) -> Dict[str, Union[int, str]]:
        def parse_summary_event(event: tf.Event) -> Tuple[str, Dict[str, str]]:
            labels = {
                "data_set": data_set_type,
                "step": event.step
            }

            scalar_non_blacklisted_summaries = {
                value.tag: f"{value.simple_value:.5f}"
                for value in event.summary.value
                if value.tag not in black_listed_tags
                if value.HasField('simple_value')
            }

            return event.step, {**labels, **scalar_non_blacklisted_summaries}

        def merge_step_summaries(step_summaries1: Tuple[int, Dict[str, Union[int, str]]],
                                 step_summaries2: Tuple[int, Dict[str, Union[int, str]]]) \
                -> Tuple[int, Dict[str, Union[int, str]]]:
            return step_summaries1[0], {**step_summaries1[1], **step_summaries2[1]}

        step_summaries = (
            parse_summary_event(event)
            for event in tf.train.summary_iterator(events_file)
            if event.HasField('summary')
        )

        summaries_by_step = [
            reduce(merge_step_summaries, summaries)
            for (step, summaries) in
            groupby(step_summaries, lambda t: t[0])  # assuming tf writes events ordered by ascending training step
        ]

        last_step_and_summary = summaries_by_step[-1]

        return last_step_and_summary[1]

    train_summary = last_step_summary_in_events(train_events_file_path, 'train')
    validation_summary = last_step_summary_in_events(validation_events_file_path, 'validation') if \
        validation_events_file_path else {}

    fieldnames = list(train_summary.keys())
    for s in validation_summary.keys():
        if s not in fieldnames:
            fieldnames.append(s)

    with open(output_file_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow(train_summary)
        if validation_summary:
            writer.writerow(validation_summary)
