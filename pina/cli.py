# pinacli_lightning_style.py
import importlib
from typing import Any, Dict
from jsonargparse import ArgumentParser, Namespace, ActionConfigFile
import yaml
from pina.problem import AbstractProblem
from pina.solver import SolverInterface
from pina.trainer import Trainer


def import_class(class_path: str):
    """Import a class from a full path string."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class PINACLI:
    REQUIRED_SECTIONS = ("trainer", "solver", "problem")

    def __init__(self, description: str = "PINA CLI"):
        self.parser = ArgumentParser(description=description, add_help=True)

        # positional stage argument (required, first)
        self.parser.add_argument(
            "stage", type=str, help="Stage to run: train/test/..."
        )

        # config file support
        self.parser.add_argument(
            "--config", help="Path to config file", action=ActionConfigFile
        )

        self.parser.add_subclass_arguments(
            baseclass=AbstractProblem,
            nested_key="problem",
            fail_untyped=False,
            instantiate=False,
        )
        self.parser.add_function_arguments(
            AbstractProblem.discretise_domain,
            nested_key="discretise_domain",
            fail_untyped=False,
        )
        self.parser.add_subclass_arguments(
            baseclass=SolverInterface,
            nested_key="solver",
            fail_untyped=False,
            instantiate=True,
        )

        self.parser.add_subclass_arguments(
            baseclass=Trainer,
            nested_key="trainer",
            fail_untyped=False,
            instantiate=False,
        )

        # parsed args / config storage
        self.args = None
        self.config = {}
        self.stage = {}

        # instantiated objects
        self.trainer = None
        self.model = None
        self.solver = None
        self.problem = None

    def parse(self):
        """Parse CLI args and config, dynamically instantiate required sections."""
        # parse everything; unknown keys are kept as dicts
        self.config = vars(self.parser.parse_args())
        self.stage = self.config["stage"]
        self._instantiate_problem()
        self._discretise_problem_domain()
        self._instantiate_solver()
        self.instanciate_trainer()

    def _instantiate_nested_classes(self, config):
        """Recursively instantiate any dict with 'class_path' inside init_args."""
        if isinstance(config, (dict, Namespace)):
            if "class_path" in config:
                return self._instantiate_class(config)
            else:
                return {
                    k: self._instantiate_nested_classes(v)
                    for k, v in config.items()
                }
        elif isinstance(config, list):
            return [self._instantiate_nested_classes(item) for item in config]
        return config

    def _instantiate_class(self, config):
        if "init_args" in config:
            config["init_args"] = self._instantiate_nested_classes(
                config["init_args"]
            )

        class_path = config["class_path"]
        init_args = config.get("init_args", {})
        class_ = import_class(class_path)
        return class_(**init_args)

    def _instantiate_problem(self):
        """Dynamically instantiate the problem class."""
        self.problem = self._instantiate_class(config=self.config["problem"])

    def _discretise_problem_domain(self):
        """Discretise the problem domain using provided arguments."""
        if "discretise_domain" in self.config:
            discretise_args = self.config.get("discretise_domain", {})
            discretise_args.pop("self", None)  # Remove 'self' if present
            self.problem.discretise_domain(**discretise_args)

    def _instantiate_solver(self):
        """Dynamically instantiate the solver class."""
        solver_cfg = self.config["solver"].as_dict()
        print(solver_cfg)
        solver_cfg["init_args"]["problem"] = self.problem
        self.solver = self._instantiate_class(solver_cfg)

    def instanciate_trainer(self):
        """Dynamically instantiate the trainer class."""
        solver_cfg = self.config["trainer"].as_dict()
        solver_cfg["init_args"]["solver"] = self.solver
        self.solver = self._instantiate_class(solver_cfg)
