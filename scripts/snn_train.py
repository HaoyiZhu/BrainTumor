# coding: utf-8
import argparse
import yaml
import os

import brain_tumor.utils.pytorch_util as ptu
from brain_tumor.utils.config_utils import (
    create_pipeline_dir,
    create_pipeline_exp_log_dir,
    create_pipeline_variant_file,
)
from brain_tumor.utils.launcher_util import (
    build_nested_variant_generator,
    run_multi_processes,
)

if __name__ == "__main__":
    # Add arguments
    parser = argparse.ArgumentParser(
        description="SNN Training"
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default="./configs/snn.yaml",
        help="experiment specification file",
    )
    parser.add_argument("-g", "--gpu", type=int, default=0, help="gpu id")
    args = parser.parse_args()

    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string, Loader=yaml.Loader)

    pipeline_dir = create_pipeline_dir(exp_specs)
    exp_specs["meta_data"]["pipeline_dir"] = pipeline_dir

    if exp_specs["meta_data"]["use_gpu"]:
        device = ptu.set_gpu_mode(True, args.gpu)
    exp_specs["meta_data"]["gpu_id"] = args.gpu

    algo_exp_specs = exp_specs["algo_training"]
    algo_exp_specs["common"] = exp_specs["common"]
    algo_exp_specs["meta_data"] = exp_specs["meta_data"]
    variants_generate_fn = build_nested_variant_generator(algo_exp_specs)

    variant_paths, algo_log_paths = [], []

    for exp_id, variant in enumerate(variants_generate_fn()):
        variant_log_dir, is_exist = create_pipeline_exp_log_dir(
            variant,
            os.path.join(
                exp_specs["meta_data"]["pipeline_dir"],
                exp_specs["common"]["model_name"],
            ),
            exp_name=variant["exp_name"],
            key_config=variant["key_config"].get("algo_training", {}),
        )
        algo_log_paths.append(variant_log_dir)
        if not is_exist:
            variant["log_dir"] = variant_log_dir
            variant_file_path = create_pipeline_variant_file(
                variant,
                exp_specs["meta_data"]["pipeline_dir"],
                exp_name=variant["exp_name"],
                exp_id=exp_id,
            )
            variant_paths.append(variant_file_path)

    run_multi_processes(
        algo_exp_specs["constants"]["script_path"],
        variant_paths,
        exp_specs["meta_data"]["num_workers"],
        exp_specs["meta_data"]["gpu_id"],
    )
