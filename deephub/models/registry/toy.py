import tensorflow as tf
from tensorflow import keras

from deephub.models import EstimatorModel, KerasModel


class DebugToyModel(EstimatorModel):

    def model_fn(self, features, labels, mode, params, config):
        if isinstance(features, dict):
            features = features[next(iter(features.keys()))]
            assert isinstance(features, tf.Tensor)

        if isinstance(labels, dict):
            labels = labels[next(iter(labels.keys()))]
            assert isinstance(labels, tf.Tensor)

        features = tf.cast(x=features, dtype=tf.float32)

        with tf.variable_scope('toy', 'trivial', [features]):
            w1 = tf.get_variable('w1', [features.get_shape()[1], params.get('hidden_neurons')],
                                 initializer=tf.constant_initializer(0.01))
            b1 = tf.get_variable('b1', [params.get('hidden_neurons')], initializer=tf.constant_initializer(0.0))
            out1 = tf.matmul(features, w1) + b1

            out1 = tf.nn.relu(out1)

            w2 = tf.get_variable('w2', [out1.get_shape()[1], params['num_classes']],
                                 initializer=tf.constant_initializer(0.01))
            b2 = tf.get_variable('b2', [params['num_classes']], initializer=tf.constant_initializer(0.0))
            out2 = tf.matmul(out1, w2) + b2

        logits = out2
        predictions = tf.argmax(logits, axis=1, name='predictions')

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions
            )

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        # loss = tf.Print(loss, [loss], 'loss: ')

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions
            )

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = params.get('optimizer', tf.train.AdamOptimizer)
            if params.get("num_steps_per_decay"):
                assert isinstance(params.get('learning_rate_decay_factor'), float)
                assert isinstance(params.get('num_steps_per_decay'), int)
                assert isinstance(params.get('learning_rate'), float)

                learning_rate = tf.train.exponential_decay(
                              params.get("learning_rate"),
                              tf.train.get_or_create_global_step(),
                              params.get('num_steps_per_decay'),
                              params.get('learning_rate_decay_factor'),
                              staircase=True)
            elif params.get('piecewise_learning_rate_schedule'):
                assert len(params['piecewise_lr_boundaries']) > 0
                assert len(params['piecewise_lr_values']) > 0

                learning_rate = tf.train.piecewise_constant(tf.train.get_or_create_global_step(),
                                                            boundaries=params['piecewise_lr_boundaries'],
                                                            values=params['piecewise_lr_values'])
            else:
                assert isinstance(params.get('learning_rate'), float)

                learning_rate = params.get("learning_rate")

            optimizer = optimizer(learning_rate=learning_rate,
                                  beta1=0.9, beta2=0.999, epsilon=1e-8)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

            training_accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))
            # training_accuracy = tf.Print(training_accuracy, [training_accuracy], 'training_accuracy: ')

            tf.summary.scalar('training_accuracy', training_accuracy)
            tf.summary.scalar("learning_rate", learning_rate)

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op
            )

        if mode == tf.estimator.ModeKeys.EVAL:
            accuracy_op = tf.metrics.accuracy(labels=labels, predictions=predictions)

            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.EVAL,
                loss=loss,
                eval_metric_ops={
                    'accuracy': accuracy_op
                }
            )


class DebugToyKerasModel(KerasModel):

    def keras_model_fn(self):
        x = keras.layers.Input(shape=(50,))
        y = keras.layers.Dense(20, name='fc20', activation='relu')(x)
        y = keras.layers.Dense(1, name='fc1', activation='sigmoid')(y)
        model = keras.Model(inputs=[x], outputs=y)

        model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
