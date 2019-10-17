import tensorflow as tf
import pytest

from deephub.models.utils import get_fine_tuning_var_list, NoTrainableVariablesError


def test_filter_trainable_variables():
    with tf.variable_scope('scope1'):
        s1_vara = tf.get_variable('vara', dtype=tf.float32, shape=(10, 2))
        s1_varb = tf.get_variable('varb', dtype=tf.float32, shape=(10, 2))

    with tf.variable_scope('scope2'):
        s2_vara = tf.get_variable('vara', dtype=tf.float32, shape=(10, 2))
        s2_varb = tf.get_variable('varb', dtype=tf.float32, shape=(10, 2))

    all_variables = [
        s1_vara, s1_varb, s2_vara, s2_varb
    ]
    assert tf.trainable_variables() == all_variables

    # Test with all variables
    assert get_fine_tuning_var_list(None) is None

    # Test with matched filtering
    assert get_fine_tuning_var_list('scope1') == [s1_vara, s1_varb]
    assert get_fine_tuning_var_list('scope2') == [s2_vara, s2_varb]

    assert get_fine_tuning_var_list('scope') == all_variables

    # Test if no variables was matched
    with pytest.raises(NoTrainableVariablesError):
        get_fine_tuning_var_list('unknown_scope')
