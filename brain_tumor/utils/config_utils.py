from typing import Dict
import os
import yaml
import json
import datetime
import dateutil
import dateutil.tz
from enum import Enum

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOCAL_LOG_DIR = os.path.join(BASE_DIR, "logs")


def get_value_with_list_keys(input_dict, list_of_keys):
    return {k: v for k, v in input_dict.items() if k in list_of_keys}


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {"$class": o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {"$enum": o.__module__ + "." + o.__class__.__name__ + "." + o.name}
        return json.JSONEncoder.default(self, o)


def traverse_one_dict_using_the_other(raw_dict: Dict, refer_dict: Dict):
    for key, value in refer_dict.items():
        if key not in raw_dict:
            continue
        if type(value) is dict:
            yield from traverse_one_dict_using_the_other(raw_dict[key], refer_dict[key])
        else:
            yield raw_dict[key], refer_dict[key]


def create_pipeline_dir(exp_specs, base_log_dir=None):
    key_configs = get_value_with_list_keys(
        exp_specs["meta_data"]["key_config"], ["common",],
    )
    exp_name = exp_specs["meta_data"]["pipeline_exp_name"]

    for value, short_name in traverse_one_dict_using_the_other(exp_specs, key_configs):
        exp_name += f"--{short_name}-{value}"
    if base_log_dir is None:
        base_log_dir = LOCAL_LOG_DIR
    pipeline_dir = os.path.join(base_log_dir, exp_name)

    if os.path.exists(pipeline_dir):
        print("WARNING: Pipeline directory already exists {}".format(pipeline_dir))
    os.makedirs(pipeline_dir, exist_ok=True)
    return pipeline_dir


def create_pipeline_variant_file(
    variant_dict: Dict, pipeline_dir: str, exp_name: str, exp_id: int = 0
):
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    variants_dir = os.path.join(
        pipeline_dir,
        "variants",
        "variants-for-" + exp_name,
        "variants-" + timestamp + "%04d" % exp_id,  # exp_id to avoid same naming
    )
    os.makedirs(variants_dir)
    variant_path = os.path.join(variants_dir, "variant.yaml")
    with open(variant_path, "w") as f:
        yaml.dump(variant_dict, f, default_flow_style=False)
    return variant_path


def create_pipeline_exp_log_dir(
    variant_dict: Dict, pipeline_dir: str, exp_name: str, key_config: Dict = None,
):
    if key_config is not None:
        for value, short_name in traverse_one_dict_using_the_other(
            variant_dict, key_config
        ):
            assert value != "seed", "Seed should not be in the key config"
            exp_name += f"--{short_name}-{value}"
    seed_folder = f"seed-{variant_dict['seed']}"
    log_dir = os.path.join(pipeline_dir, exp_name, seed_folder)
    if os.path.exists(log_dir):
        is_exist = True
    else:
        os.makedirs(log_dir)
        with open(os.path.join(log_dir, "variant.json"), "w") as f:
            json.dump(variant_dict, f, indent=2, sort_keys=True, cls=MyEncoder)
        is_exist = False
    return log_dir, is_exist
