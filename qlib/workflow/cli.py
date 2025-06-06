#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
import logging
import os
from pathlib import Path
import sys

import fire, wandb
from jinja2 import Template, meta
from ruamel.yaml import YAML

import qlib
from qlib.config import C
from qlib.log import get_module_logger
from qlib.model.trainer import task_train
from qlib.utils import set_log_with_config
from qlib.utils.data import update_config

set_log_with_config(C.logging_config)
logger = get_module_logger("qrun", logging.INFO)


def get_path_list(path):
    if isinstance(path, str):
        return [path]
    else:
        return list(path)


def sys_config(config, config_path):
    """
    Configure the `sys` section

    Parameters
    ----------
    config : dict
        configuration of the workflow.
    config_path : str
        path of the configuration
    """
    sys_config = config.get("sys", {})

    # abspath
    for p in get_path_list(sys_config.get("path", [])):
        sys.path.append(p)

    # relative path to config path
    for p in get_path_list(sys_config.get("rel_path", [])):
        sys.path.append(str(Path(config_path).parent.resolve().absolute() / p))


def render_template(config_path: str) -> str:
    """
    render the template based on the environment

    Parameters
    ----------
    config_path : str
        configuration path

    Returns
    -------
    str
        the rendered content
    """
    with open(config_path, "r") as f:
        config = f.read()
    # Set up the Jinja2 environment
    template = Template(config)

    # Parse the template to find undeclared variables
    env = template.environment
    parsed_content = env.parse(config)
    variables = meta.find_undeclared_variables(parsed_content)

    # Get context from os.environ according to the variables
    context = {var: os.getenv(var, "") for var in variables if var in os.environ}
    logger.info(f"Render the template with the context: {context}")

    # Render the template with the context
    rendered_content = template.render(context)
    return rendered_content


# workflow handler function
def workflow(config_path, experiment_name="workflow", uri_folder="mlruns"):
    """
    This is a Qlib CLI entrance.
    User can run the whole Quant research workflow defined by a configure file
    - the code is located here ``qlib/workflow/cli.py`

    User can specify a base_config file in your workflow.yml file by adding "BASE_CONFIG_PATH".
    Qlib will load the configuration in BASE_CONFIG_PATH first, and the user only needs to update the custom fields
    in their own workflow.yml file.

    For examples:

        qlib_init:
            provider_uri: "~/.qlib/qlib_data/cn_data"
            region: cn
        BASE_CONFIG_PATH: "workflow_config_lightgbm_Alpha158_csi500.yaml"
        market: csi300

    """
    # Render the template (load config from yaml file)
    rendered_yaml = render_template(config_path)
    yaml = YAML(typ="safe", pure=True)
    config = yaml.load(rendered_yaml)

    base_config_path = config.get("BASE_CONFIG_PATH", None)
    if base_config_path:
        logger.info(f"Use BASE_CONFIG_PATH: {base_config_path}")
        base_config_path = Path(base_config_path)

        # it will find config file in absolute path and relative path
        if base_config_path.exists():
            path = base_config_path
        else:
            logger.info(
                f"Can't find BASE_CONFIG_PATH base on: {Path.cwd()}, "
                f"try using relative path to config path: {Path(config_path).absolute()}"
            )
            relative_path = Path(config_path).absolute().parent.joinpath(base_config_path)
            if relative_path.exists():
                path = relative_path
            else:
                raise FileNotFoundError(f"Can't find the BASE_CONFIG file: {base_config_path}")

        with open(path) as fp:
            yaml = YAML(typ="safe", pure=True)
            base_config = yaml.load(fp)
        logger.info(f"Load BASE_CONFIG_PATH succeed: {path.resolve()}")
        config = update_config(base_config, config)

    # config the `sys` section
    sys_config(config, config_path)

    if config["task"]["model"]["kwargs"].get("wandb", False):
        wandb.init(project="qlib", dir=str(Path(os.getcwd()).resolve() / uri_folder), config=config, name=config["wandb_name"])

    if "exp_manager" in config.get("qlib_init"):
        qlib.init(**config.get("qlib_init"))
    else: # default manager
        exp_manager = C["exp_manager"]
        exp_manager["kwargs"]["uri"] = "file:" + str(Path(os.getcwd()).resolve() / uri_folder)
        qlib.init(**config.get("qlib_init"), exp_manager=exp_manager)

    if "experiment_name" in config:
        experiment_name = config["experiment_name"]
    recorder = task_train(config.get("task"), experiment_name=experiment_name) # train, eval and test
    recorder.save_objects(config=config)


# function to run workflow by config
def run():
    fire.Fire(workflow)


if __name__ == "__main__":
    run()
