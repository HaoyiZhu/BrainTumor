import inspect
import numpy as np
import json
import os
import os.path as osp
import datetime
import dateutil.tz
from time import sleep
from subprocess import Popen
from brain_tumor.utils import logger
from brain_tumor.utils.copy import deepcopy
from brain_tumor.utils.errors import ExperimentError

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOCAL_LOG_DIR = os.path.join(BASE_DIR, "languagemodel", "logs")


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class VariantDict(AttrDict):
    def __init__(self, d, hidden_keys):
        super(VariantDict, self).__init__(d)
        self._hidden_keys = hidden_keys

    def dump(self):
        return {k: v for k, v in self.items() if k not in self._hidden_keys}


class VariantGenerator(object):
    """
    Usage:

    vg = VariantGenerator()
    vg.add("param1", [1, 2, 3])
    vg.add("param2", ['x', 'y'])
    vg.variants() => # all combinations of [1,2,3] x ['x','y']

    Supports noncyclic dependency among parameters:
    vg = VariantGenerator()
    vg.add("param1", [1, 2, 3])
    vg.add("param2", lambda param1: [param1+1, param1+2])
    vg.variants() => # ..
    """

    def __init__(self):
        self._variants = []
        self._populate_variants()
        self._hidden_keys = []
        for k, vs, cfg in self._variants:
            if cfg.get("hide", False):
                self._hidden_keys.append(k)

    def add(self, key, vals, **kwargs):
        self._variants.append((key, vals, kwargs))

    def _populate_variants(self):
        methods = inspect.getmembers(
            self.__class__,
            predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x),
        )
        methods = [
            x[1].__get__(self, self.__class__)
            for x in methods
            if getattr(x[1], "__is_variant", False)
        ]
        for m in methods:
            self.add(m.__name__, m, **getattr(m, "__variant_config", dict()))

    def variants(self, randomized=False):
        ret = list(self.ivariants())
        if randomized:
            np.random.shuffle(ret)
        return list(map(self.variant_dict, ret))

    def variant_dict(self, variant):
        return VariantDict(variant, self._hidden_keys)

    def to_name_suffix(self, variant):
        suffix = []
        for k, vs, cfg in self._variants:
            if not cfg.get("hide", False):
                suffix.append(k + "_" + str(variant[k]))
        return "_".join(suffix)

    def ivariants(self):
        dependencies = list()
        for key, vals, _ in self._variants:
            if hasattr(vals, "__call__"):
                args = inspect.getargspec(vals).args
                if hasattr(vals, "im_self") or hasattr(vals, "__self__"):
                    # remove the first 'self' parameter
                    args = args[1:]
                dependencies.append((key, set(args)))
            else:
                dependencies.append((key, set()))
        sorted_keys = []
        # topo sort all nodes
        while len(sorted_keys) < len(self._variants):
            # get all nodes with zero in-degree
            free_nodes = [k for k, v in dependencies if len(v) == 0]
            if len(free_nodes) == 0:
                error_msg = "Invalid parameter dependency: \n"
                for k, v in dependencies:
                    if len(v) > 0:
                        error_msg += k + " depends on " + " & ".join(v) + "\n"
                raise ValueError(error_msg)
            dependencies = [(k, v) for k, v in dependencies if k not in free_nodes]
            # remove the free nodes from the remaining dependencies
            for _, v in dependencies:
                v.difference_update(free_nodes)
            sorted_keys += free_nodes
        return self._ivariants_sorted(sorted_keys)

    def _ivariants_sorted(self, sorted_keys):
        if len(sorted_keys) == 0:
            yield dict()
        else:
            first_keys = sorted_keys[:-1]
            first_variants = self._ivariants_sorted(first_keys)
            last_key = sorted_keys[-1]
            last_vals = [v for k, v, _ in self._variants if k == last_key][0]
            if hasattr(last_vals, "__call__"):
                last_val_keys = inspect.getargspec(last_vals).args
                if hasattr(last_vals, "im_self") or hasattr(last_vals, "__self__"):
                    last_val_keys = last_val_keys[1:]
            else:
                last_val_keys = None
            for variant in first_variants:
                if hasattr(last_vals, "__call__"):
                    last_variants = last_vals(**{k: variant[k] for k in last_val_keys})
                    for last_choice in last_variants:
                        yield AttrDict(variant, **{last_key: last_choice})
                else:
                    for last_choice in last_vals:
                        yield AttrDict(variant, **{last_key: last_choice})


def check_exp_spec_format(specs):
    """
    Check that all keys are strings that don't contain '.'
    """
    for k, v in specs.items():
        if not isinstance(k, str):
            return False
        if "." in k:
            return False
        if isinstance(v, dict):
            sub_ok = check_exp_spec_format(v)
            if not sub_ok:
                return False
    return True


def flatten_dict(dic):
    """
    Assumes a potentially nested dictionary where all keys
    are strings that do not contain a '.'

    Returns a flat dict with keys having format:
    {'key.sub_key.sub_sub_key': ..., etc.}
    """
    new_dic = {}
    for k, v in dic.items():
        if isinstance(v, dict):
            sub_dict = flatten_dict(v)
            for sub_k, v in sub_dict.items():
                new_dic[".".join([k, sub_k])] = v
        else:
            new_dic[k] = v

    return new_dic


def add_variable_to_constant_specs(constants, flat_variables):
    new_dict = deepcopy(constants)
    for k, v in flat_variables.items():
        cur_sub_dict = new_dict
        split_k = k.split(".")
        for sub_key in split_k[:-1]:
            cur_sub_dict = cur_sub_dict[sub_key]
        cur_sub_dict[split_k[-1]] = v
    return new_dict


def build_nested_variant_generator(exp_spec):
    assert check_exp_spec_format(exp_spec)
    # from rllab.misc.instrument import VariantGenerator

    variables = exp_spec["variables"]
    constants = exp_spec["constants"]

    # check if we're effectively just running a single experiment
    if variables is None:

        def vg_fn():
            dict_to_yield = constants
            dict_to_yield.update(exp_spec["meta_data"])
            yield dict_to_yield

        return vg_fn

    variables = flatten_dict(variables)
    vg = VariantGenerator()
    for k, v in variables.items():
        vg.add(k, v)

    def vg_fn():
        for flat_variables in vg.variants():
            dict_to_yield = add_variable_to_constant_specs(constants, flat_variables)
            dict_to_yield.update(exp_spec["meta_data"])
            if "common" in exp_spec:
                dict_to_yield.update(exp_spec["common"])
            del dict_to_yield["_hidden_keys"]
            yield dict_to_yield

    return vg_fn


def run_single_process(script_path, variant_path, gpu_id):
    command = "python {} -e {} -g {}".format(script_path, variant_path, gpu_id,)
    ret_code = os.system(command)
    if ret_code != 0:
        raise ExperimentError(script_path, [variant_path])


def run_multi_processes(script_path, variant_paths, num_workers, gpu_id):
    if len(variant_paths) < 0:
        return

    num_variants = len(variant_paths)
    num_workers = min(num_workers, num_variants)
    command = "python {script_path} -e {specs} -g {gpu_id}"

    running_processes = []
    command_format_dicts = []
    args_idx = 0
    is_terminated = False
    try:
        while args_idx < num_variants or len(running_processes) > 0:
            if len(running_processes) < num_workers and args_idx < num_variants:
                command_format_dict = {}
                command_format_dict["script_path"] = script_path
                command_format_dict["specs"] = variant_paths[args_idx]
                command_format_dict["gpu_id"] = gpu_id

                command_to_run = command.format(**command_format_dict)
                command_to_run = command_to_run.split()
                print(command_to_run)
                p = Popen(command_to_run)
                args_idx += 1
                running_processes.append(p)
                command_format_dicts.append(command_format_dict)
            else:
                sleep(1)

            new_running_processes = []
            new_command_format_dicts = []
            for d, p in zip(command_format_dicts, running_processes):
                ret_code = p.poll()
                if ret_code is None or ret_code != 0:
                    new_running_processes.append(p)
                    new_command_format_dicts.append(d)
                    if ret_code is not None:
                        is_terminated = True
            running_processes = new_running_processes
            command_format_dicts = new_command_format_dicts

            if is_terminated:
                break

    except KeyboardInterrupt:
        is_terminated = True

    if is_terminated:
        for p in running_processes:
            p.terminate()

        # terminate currently running experiments and not-yet runned experiments
        raise ExperimentError(
            script_path,
            [*[d["specs"] for d in command_format_dicts], *variant_paths[args_idx:],],
        )


def create_exp_name(exp_prefix, exp_id=0, seed=0):
    """
    Create a semi-unique experiment name that has a timestamp
    :param exp_prefix:
    :param exp_id:
    :return:
    """
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    return "%s_%s_%04d--s-%d" % (exp_prefix, timestamp, exp_id, seed)


def create_log_dir(exp_prefix, exp_id=0, seed=0, base_log_dir=None):
    """
    Creates and returns a unique log directory.

    :param exp_prefix: All experiments with this prefix will have log
    directories be under this directory.
    :param exp_id: Different exp_ids will be in different directories.
    :return:
    """
    exp_name = create_exp_name(exp_prefix, exp_id=exp_id, seed=seed)
    if base_log_dir is None:
        base_log_dir = LOCAL_LOG_DIR
    log_dir = osp.join(base_log_dir, exp_prefix.replace("_", "-"), exp_name)
    if osp.exists(log_dir):
        print("WARNING: Log directory already exists {}".format(log_dir))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def dict_to_safe_json(d):
    """
    Convert each value in the dictionary into a JSON'able primitive.
    :param d:
    :return:
    """
    new_d = {}
    for key, item in d.items():
        if safe_json(item):
            new_d[key] = item
        else:
            if isinstance(item, dict):
                new_d[key] = dict_to_safe_json(item)
            else:
                new_d[key] = str(item)
    return new_d


def safe_json(data):
    if data is None:
        return True
    elif isinstance(data, (bool, int, float)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(safe_json(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and safe_json(v) for k, v in data.items())
    return False


def setup_logger(
    exp_prefix="default",
    exp_id=0,
    seed=0,
    variant=None,
    base_log_dir=None,
    text_log_file="debug.log",
    variant_log_file="variant.json",
    tabular_log_file="progress.csv",
    snapshot_mode="last",
    snapshot_gap=1,
    log_wandb=True,
    log_tabular_only=False,
    log_dir=None,
    git_info=None,
    script_name=None,
):
    """
    Set up logger to have some reasonable default settings.

    Will save log output to

        based_log_dir/exp_prefix/exp_name.

    exp_name will be auto-generated to be unique.

    If log_dir is specified, then that directory is used as the output dir.

    :param exp_prefix: The sub-directory for this specific experiment.
    :param exp_id: The number of the specific experiment run within this
    experiment.
    :param variant:
    :param base_log_dir: The directory where all log should be saved.
    :param text_log_file:
    :param variant_log_file:
    :param tabular_log_file:
    :param snapshot_mode:
    :param log_tabular_only:
    :param snapshot_gap:
    :param log_dir:
    :param git_info:
    :param script_name: If set, save the script name to this.
    :return:
    """
    first_time = log_dir is None
    if first_time:
        log_dir = create_log_dir(
            exp_prefix, exp_id=exp_id, seed=seed, base_log_dir=base_log_dir
        )

    if variant is not None:
        logger.log("Variant:")
        logger.log(json.dumps(dict_to_safe_json(variant), indent=2))
        variant_log_path = osp.join(log_dir, variant_log_file)
        logger.log_variant(variant_log_path, variant)

    tabular_log_path = osp.join(log_dir, tabular_log_file)
    text_log_path = osp.join(log_dir, text_log_file)

    logger.add_text_output(text_log_path)
    if first_time:
        logger.add_tabular_output(tabular_log_path)
    else:
        logger._add_output(
            tabular_log_path, logger._tabular_outputs, logger._tabular_fds, mode="a"
        )
        for tabular_fd in logger._tabular_fds:
            logger._tabular_header_written.add(tabular_fd)
    logger.set_snapshot_dir(log_dir, variant, log_wandb)
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_snapshot_gap(snapshot_gap)
    logger.set_log_tabular_only(log_tabular_only)
    exp_name = log_dir.split("/")[-1]
    logger.push_prefix("[%s] " % exp_name)

    if git_info is not None:
        code_diff, commit_hash, branch_name = git_info
        if code_diff is not None:
            with open(osp.join(log_dir, "code.diff"), "w") as f:
                f.write(code_diff)
        with open(osp.join(log_dir, "git_info.txt"), "w") as f:
            f.write("git hash: {}".format(commit_hash))
            f.write("\n")
            f.write("git branch name: {}".format(branch_name))
    if script_name is not None:
        with open(osp.join(log_dir, "script_name.txt"), "w") as f:
            f.write(script_name)
    return log_dir
