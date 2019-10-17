## Training Procedure

#### Train a model
Deephub framework is using Tensorflow's Estimators in order to execute the training procedure. You must 
specify the model_dir parameter in order to handle the output files folder, and two files for the training and 
validation data. The model_dir folder contains all the information needed for functionalities like 
visualization of model's performance through Tensorboard, warm-starting, serving etc.

Using the `-p` parameter on `trainer` cli entry point you can overwrite any parameter you wish.

The training functionality can be invoked through the CLI as:
```
$ deep -l DEBUG trainer train MODEL_NAME 
-p model.model_dir /path_to_model_dir \
-p train.train_feeder.file_patterns /path_to_training_file_patterns \ 
-p train.eval_feeder.file_patterns /path_to_validation_file_patterns \
```

Please use the `--help` option to see detailed description about the usage.

#### Train a dummy network example
You can instantly invoke a training on toy dataset for testing and debugging purposes through the CLI.
As an example, to train a `toy` network using a toy dataset you run the following:
```
$ deep -l DEBUG trainer train toy \
-p model.model_dir /AnyPath \
-p train.train_feeder.file_patterns deephub/resources/blobs/data/toy/train-* \ 
-c train.eval_feeder.file_patterns deephub/resources/blobs/data/toy/validation-* \
```

For more information about the parameters like gpu pinning etc you can use the `--help` option.

##### Experiment configuation
In order for trainer to be able to execute any experiment, there must exists a valid configuration for such a model,
specified correctly in `resources/blobs/config/variants/` folder. You can have a look into this file, 
for the toy model configuration, in order to have a clear understanding about the objects needed to be specified in
order for the trainer to be able to execute an end to end experiment.

###### Specify model parameters
As you can see in `toy.yaml` file we have to declare this variant's model parameters. 
We are using `metayaml` package for reading yaml variants so this allow to do various nice things like: extending
the base yaml, overwrites, inheritace and other. An example for such a configuration for this toy model is 
```
model:
  type: 'toy:DebugToyModel'
  num_classes: 100
  learning_rate: 0.001
  hidden_neurons: 50
train:
  epochs: 10
  save_checkpoint_steps: 100
#      save_checkpoint_secs: 1
  validation_secs: 3600
  save_summary_steps: 1
extend:
  - ../feeders/feeder_toy_train.yaml
  - ../feeders/feeder_toy_eval.yaml
```
The type of the model param, specifies in which file to search for the appropriate network class definition. In
the example above, search into `toy.py` inside `models/registry/` folder for the `DebugToyModel` class network. 
As model params you can declare anything your model needs to know.

###### Specify train parameters
After model params, you have to specify specific parameters for the training procedure. For the toy example above 
```
  train:
    epochs: 10
    validation_secs: 3600
    save_summary_steps: 1
```
To make these parameters clear, one needs to know that `save_summary_steps` controls how ofter the Tensorflow Estimator
will save α checkpoint during training. The eval params controls how often the Tensorflow Estimator will execute the
validation procedure during training. For now the framework keeps the minimum of these two parameters, and sets this
value on Estimator's save_checkpoint_secs param.

###### Sepcify training and (optionally) eval feeders
The next step is to specify the right ETL for this model. For the toy experiment above
```
    train_feeder:
      type: 'TFRecordExamplesFeeder'
      batch_size: 10
      shuffle: False
      file_patterns: ['']
      labels_map:
        number:
          type: FixedLenFeature
          shape: [1]
          dtype: int64
      features_map:
        number:
          type: FixedLenFeature
          shape: [1]
          dtype: int64
      eval_feeder:
      type: 'TFRecordExamplesFeeder'
      batch_size: 10
      shuffle: False
      file_patterns: ['']
      labels_map:
        number:
          type: FixedLenFeature
          shape: [1]
          dtype: int64
      features_map:
        number:
          type: FixedLenFeature
          shape: [1]
          dtype: int64
```
With the above configuration we are specifying a TFRecordExamplesFeeder (which is a framework's specific implementation 
in order to read TFrecord files). We are also specifying that these features is a mapping in the form of 
`{'number': tf.io.FixedLenFeature(shape=([1]), dtype=tf.int64)}`.

With such a complete configuration, the trainer has all the information needed to run an end to end experiment.
 

#### Important note
For a TFRecords dataset the framework is expecting a custom defined _tfrecord_metadata.json file. This file is
important in order to keep track of the files of a dataset and to detect possible changes into its files. You can
generate such a file using `utils` entry point. An example for such a command is
```
$ deep utils generate-metadata '/path-to-files-folder/*'
```
