import pytest

import tensorflow as tf
import numpy as np
import os

from deephub.models.registry.toy import DebugToyModel
from deephub.models.feeders import MemorySamplesFeeder
from deephub.trainer import Trainer


@pytest.mark.slow
def test_adaptive_learning_rate(tmpdir):
    model_params = {
        'type': 'toy:DebugToyModel',
        'model_dir': str(tmpdir / 'test_adaptive_learning_rate/'),
        'learning_rate': 0.1,
        'num_steps_per_decay': 2,
        'learning_rate_decay_factor': 0.1,
        'num_classes': 2,
        'num_steps_per_epoch': 1,
        'hidden_neurons': 10
    }

    train_params = {
        'epochs': 10,
        'save_summary_steps': 1,
        'save_checkpoint_steps': 1
    }

    assert model_params['learning_rate'] == 0.1
    assert model_params['num_steps_per_decay'] == 2
    assert model_params['learning_rate_decay_factor'] == 0.1
    assert train_params['epochs'] == 10

    # Create the ground truth learning rate values
    ground_trouth_learning_rate_values = [0.1, 0.1, 0.01, 0.01, 0.001, 0.001, 0.0001, 0.0001, 0.00001, 0.00001]

    # Initialize the model
    model = DebugToyModel(**model_params)

    # Read training data Be careful for batch_size == Number of examples, because it affects num_steps_per_epoch
    train_feeder = MemorySamplesFeeder(np.asarray(np.arange(0, 10).reshape(10, 1), dtype=np.float32),
                                       np.array([int(0 if i < 5 else 1)
                                                for i in range(10)], dtype=np.int64, ndmin=1),
                                       batch_size=10, feed_as_dict=False)

    # Initialize the Trainer
    trainer = Trainer()

    # Start training process
    trainer.train(model=model, train_feeder=train_feeder, eval_feeder=None, **train_params)

    learning_rate_values = list()
    # Parse events out tensorboard file
    events_out_file = os.path.join(model_params['model_dir'],
                                   [str(f) for f in os.listdir(model_params['model_dir'])
                                    if f.startswith('events.out.tfevents')][0])
    for e in tf.train.summary_iterator(events_out_file):
        for v in e.summary.value:
            if v.tag == 'learning_rate':
                learning_rate_values.append(float('{:.7f}' .format(v.simple_value)))

    # Testing condition
    assert ground_trouth_learning_rate_values == learning_rate_values
