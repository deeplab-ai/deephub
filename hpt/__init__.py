from typing import Dict, Optional, List, Any, Callable
from enum import Enum, unique
import numpy as np
from hyperopt import fmin, tpe, hp

from deephub.common.io import load_yaml, AnyPathType
from deephub.common.utils import hash_from_dict
from pathlib import Path




class HptParamSpec:
    """
    HPT parameter specifications.
    """

    @unique
    class ParamType(Enum):
        """Enumeration of different type of parameters"""
        INTEGER = 0
        REAL = 1
        CATEGORICAL = 2

    @unique
    class Scale(Enum):
        """Enumeration of different type of scaling"""
        NO_SCALE = 0
        LOG = 1

    def __init__(self,
                 name: str,
                 type: ParamType,
                 constant: Any = None,
                 min: Optional[float] = None,
                 max: Optional[float] = None,
                 choices: Optional[List] = None,
                 scale: Scale = Scale.NO_SCALE):
        """
        Create a new parameter specification.

        :param name: The unique name of parameter
        :param type: The type of the parameter. Only values from ParamType are accepted here
        :param constant: Set this to a value if you want this parameter to have a fixed value. If make it a
            constant you cannot set :min:, max, or categories.
        :param min: The minimum value that the variable can take. For integers this is inclusive. Setting a range
            requires both @min and @max to be set.
        :param max: The maximum value that the variable can take. For integers this is inclusive. Setting a range
            requires both @min and @max to be set.
        :param choices: The different choices if it is a categorical variable.
        :param scale: The scaling of the sampled space. Only values from Scale enumeration are accepted here
        """

        def assure_min_max():
            if min is None or max is None:
                raise ValueError(f"You need to provide minimum and maximum values for parameter '{name}'")

        def assure_constant():
            if constant is None:
                raise ValueError(
                    f"You need to provide a constant for parameter '{name}'")

        def assure_choices():
            if choices is None:
                raise ValueError(
                    f"You need to define choices for parameter '{name}'")

        def assure_constant_or_range():
            try:
                assure_min_max()
            except:
                assure_constant()

        # Check input
        if type in {self.ParamType.INTEGER, self.ParamType.REAL}:
            assure_constant_or_range()
        elif type == self.ParamType.CATEGORICAL:
            assure_choices()
        else:
            raise ValueError(f"Unknown parameter type '{self.param_type}'")

        self.name = name
        self.param_type = type
        self.constant = constant
        self.min = min
        self.max = max
        self.choices = choices
        self.scale = scale

    @property
    def is_constant(self) -> bool:
        """Check if this parameter is a constant"""
        return self.constant is not None

    def create_hp_variable(self):
        """
        Create an hyperopt variable object based on the specifications of the variable
        :return: The newly created object
        """

        if self.is_constant:
            return self.constant

        if self.param_type == self.ParamType.CATEGORICAL:
            return hp.choice(self.name, self.choices)
        elif self.param_type == self.param_type.INTEGER:
            return self.min + hp.randint(self.name, self.max - self.min + 1)
        elif self.param_type == self.param_type.REAL:
            if self.scale == self.Scale.LOG:
                return hp.loguniform(self.name, np.log(self.min), np.log(self.max))
            elif self.scale == self.Scale.NO_SCALE:
                return hp.uniform(self.name, self.min, self.max)


class ModuleCaller:
    """
    A special callable object that can run any arbitrary function from any module

    Example:

    >>> train = ModuleCaller('models.goodone', 'train')
    >>> train(epochs=1)

    """

    def __init__(self, module: str, function: str):
        """
        Create a new caller object
        :param module: The module path in dotted notation. e.g. foo.bar Make sure to give an absolute and meaningfull
            path in respect of the execution environment
        :param function: The name of the function to import from module and call
        """
        self.module = module
        self.function_name = function

    @property
    def function(self) -> Callable:
        """The actual referenced function object."""
        local_space = {}
        exec(f"from {self.module} import {self.function_name} as function", globals(), local_space)

        return local_space['function']

    def __call__(self, *args, **kwargs) -> Any:
        """
        Call function with position and keyword arguments
        """
        return self.function(*args, **kwargs)


class TuningPlan:
    """
    A tuning plan that can be executed to optimize hyper parameters.

    The plan can be designed through API or loaded from a YAML file using `load_from_file()` class method.
    """

    @unique
    class Goal(Enum):
        MINIMIZE = 0
        MAXIMIZE = 1

    def __init__(self,
                 goal: Goal,
                 objective: Callable,
                 params_specs: List[HptParamSpec],
                 metric: Optional[str] = None,
                 max_trials: Optional[int] = None,
                 trial_id_argument: Optional[str] = None,
                 ):
        """
        Create a tuning plan
        :param goal: The goal of optimization i.e. minimize or maximize. Use one of Goal enumeration values.
        :param objective: The callable to evaluate a parameter configuration
        :param params_specs: A list with specifications of all parameters that will be tuned.
        :param hpt_folder: The path to save results for each trial
        :param metric: The objective should return a scalar with the loss. If more values are returned in a dictionary
            this is the key to the dictionary.
        :param max_trials: The maximum number of trials to run.
        :param trial_id_argument: An argument of objective function to pass a unique id of the hpt call
        """

        self.goal = goal
        self.objective = objective
        self.metric = metric
        self.params_specs = params_specs
        self.max_trials = max_trials
        self.trial_id_argument = trial_id_argument

        # self.hpt_folder = hpt_folder

    @classmethod
    def load_from_file(cls, fname: AnyPathType) -> 'TuningPlan':
        """
        Load a tuning plan from a YAML file
        :param fname: The path to the file
        :return: An initialized and not executed tuning plan object
        """
        if not isinstance(fname, Path):
            fname = Path(fname)

        configuration = load_yaml(fname)

        objective = configuration['objective']
        if objective['type'] == 'MODULE_CALL':
            objective = ModuleCaller(module=objective['module'], function=objective['function'])
        else:
            raise ValueError(f"Unknown call type {type}")

        # Parse param specs
        params_specs = []
        for name, specs in configuration['params'].items():
            specs['type'] = HptParamSpec.ParamType[specs['type'].upper()]
            if 'scale' in specs:
                specs['scale'] = HptParamSpec.Scale[specs['scale'].upper()]

            params_specs.append(HptParamSpec(name=name, **specs))

        return cls(
            goal=cls.Goal[configuration['goal'].upper()],
            objective=objective,
            params_specs=params_specs,
            metric=configuration.get('metric'),
            max_trials=configuration.get('max_trials'),
            trial_id_argument=configuration.get('trial_id_argument'),


        )

    def _create_space(self) -> Dict:
        """Create the hyperopt space"""
        return {
            param_specs.name: param_specs.create_hp_variable()
            for param_specs in self.params_specs
        }

    def _call_and_get_metric(self, **kwargs):
        """Wrapper to call objective and get the correct loss metric"""

        if self.trial_id_argument is not None:
            kwargs[self.trial_id_argument] = hash_from_dict(kwargs)
        results = self.objective(**kwargs)

        if self.metric is not None:
            loss = results[self.metric]
            if isinstance(loss, list):
                if self.goal == self.Goal.MAXIMIZE:
                    loss = float(max(loss))
                else:
                    loss = float(min(loss))
        elif isinstance(self.metric, dict):
            raise RuntimeError("The loss function returned a dictionary instead of scalar value. You should "
                               "use `metric` in .yaml file.")
        else:
            loss = results

        if self.goal == self.Goal.MAXIMIZE:
            return -loss
        else:
            return loss

    def optimize(self, **extra_arguments) -> Dict:
        """
        Execute the plan and optimize hyper parameters
        :param extra_arguments: Fixed arguments for optimization function
        :return: A dictionary with the best configuration values
        """

        best = fmin(fn=lambda kwargs: self._call_and_get_metric(**{**kwargs, **extra_arguments}),
                    space=self._create_space(),
                    algo=tpe.suggest,
                    max_evals=self.max_trials)
        return best