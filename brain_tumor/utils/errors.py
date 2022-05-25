from typing import List


class ExperimentError(Exception):
    def __init__(self, script_path: str, variant_paths: List[str]):
        super().__init__(script_path, variant_paths)
        self.script_path = script_path
        self.variant_paths = variant_paths

    def __str__(self):
        return "ExperimentError: Terminate during executing '{}'".format(
            self.script_path
        )
