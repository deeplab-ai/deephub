import pytest

import numpy as np

from deephub.models.registry.toy import DebugToyModel
from deephub.models.feeders import MemorySamplesFeeder
from deephub.trainer import Trainer


@pytest.mark.slow
def test_early_stopping(tmpdir):
    model_params = {
        'type': 'toy:DebugToyModel',
        'model_dir': str(tmpdir / 'test_early_stopping/'),
        'learning_rate': 0.01,
        'num_classes': 2,
        'num_steps_per_epoch': 1,
        'hidden_neurons': 512
    }

    # Initialize the model
    model = DebugToyModel(**model_params)

    # Read training data
    train_feeder = MemorySamplesFeeder(np.asarray(np.arange(0, 10).reshape(10, 1), dtype=np.float32),
                                       np.array([int(0 if i < 5 else 1)
                                                for i in range(10)], dtype=np.int64, ndmin=1),
                                       batch_size=10, feed_as_dict=False)

    # Read validation data
    validation_feeder = MemorySamplesFeeder(np.asarray(np.arange(10, 20).reshape(10, 1), dtype=np.float32),
                                            np.array([int(0 if i < 5 else 1)
                                                     for i in range(10)], dtype=np.int64, ndmin=1),
                                            batch_size=10, feed_as_dict=False)

    # Initialize the Trainer
    trainer = Trainer()
    train_params = {
        'epochs': 12,
        'save_summary_steps': 1,
        'save_checkpoint_steps': 1,
        'early_stopping_metric': 'loss',
        'early_stopping_steps_without_decrease': 3,
        'early_stopping_min_steps': 1,
        'early_stopping_hook_run_every_steps': 1
    }

    # Start training process
    trainer.train(model=model, train_feeder=train_feeder, eval_feeder=validation_feeder,
                  **train_params)
    # Grab global step from model.estimator object
    global_step = model.estimator().get_variable_value('global_step')

    assert global_step < train_params['epochs']
