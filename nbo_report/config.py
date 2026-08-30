"""
Config loading, path resolution, and the shim that lets this package
import preprocess.py and model.py out of their own directories.

THE PATH RULE
-------------
A relative path inside a config file resolves against the directory
holding that config file. That single rule keeps three things true at
once:

    cd nbo_data_preprocessing && python preprocess.py    still works
    python eda.py                                        works from root
    nothing has to be an absolute path checked into git

REUSE, NOT REIMPLEMENTATION
---------------------------
The stage scripts are imported and called, never copied. In particular
the sequence analysis calls preprocess.load_data / split_outcome_rows /
compute_anchors, so it inherits the real anchor rule -- and therefore
stays on the correct side of the leakage invariant -- instead of
re-deriving a cut-off of its own and getting it subtly wrong.

Running a whole stage is a subprocess call against the real script with
an explicit --config, so what the report describes is exactly what the
command line would have done. Importing a function to reuse it and
shelling out to run a stage are different jobs; this module does both.
"""

import importlib.util
import os
import subprocess
import sys

import yaml

# Repo root is the parent of this package. Everything else hangs off it.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =====================================================================
# paths and yaml
# =====================================================================

def resolve(path, base):
    """Resolve `path` against `base` unless it is already absolute."""
    if path is None:
        return None
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base, path))


def load_yaml(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_stage_config(path):
    """
    Load a stage config and rewrite its `io:` paths to absolute, using
    that config's own directory as the base. Returns
    (cfg, config_path, config_dir) so callers can still hand the
    original path to the script.
    """
    path = resolve(path, ROOT)
    base = os.path.dirname(path)
    cfg = load_yaml(path)
    for key, value in list(cfg.get("io", {}).items()):
        if isinstance(value, str):
            cfg["io"][key] = resolve(value, base)
    return cfg, path, base


def _looks_like_path(key):
    """io: entries naming a bare filename, not a path, stay untouched."""
    return not key.endswith(("report", "list", "profile"))


class ReportContext(object):
    """
    Everything the report modules need, loaded once.

        cfg         report_config.yaml, io paths made absolute
        pre         nbo_data_preprocessing/config.yaml, io absolute
        mdl         nbo_data_modeling/model_config.yaml, io absolute
        pre_path    original path, for handing to the script
        mdl_path    same, for the modelling stage
    """

    def __init__(self, report_config_path):
        report_config_path = resolve(report_config_path, ROOT)
        self.report_config_path = report_config_path
        base = os.path.dirname(report_config_path)
        self.cfg = load_yaml(report_config_path)

        for key, value in list(self.cfg["io"].items()):
            if _looks_like_path(key):
                self.cfg["io"][key] = resolve(value, base)

        self.pre, self.pre_path, self.pre_dir = load_stage_config(
            self.cfg["io"]["preprocess_config"])
        self.mdl, self.mdl_path, self.mdl_dir = load_stage_config(
            self.cfg["io"]["model_config"])

    # ---- convenience accessors --------------------------------------

    @property
    def schema(self):
        """
        Structural column names preprocess.py writes. Declared in
        report_config.yaml so no reporting module contains a literal.
        """
        return self.cfg["schema"]

    @property
    def labels(self):
        """Vocabulary for generated prose (converter, conversion, ...)."""
        return self.cfg["labels"]

    def bookkeeping_columns(self, design):
        """
        Columns that exist in the dataset but are keys, labels or
        bookkeeping -- never features, never profiled as features.
        """
        schema = self.schema
        names = {schema["customer"], self.label_column(design)}
        if design == "flat":
            names.add(schema["anchor_date"])
        else:
            names.update(schema["hazard_bookkeeping"])
        return names

    def report_path(self, key):
        """A file inside io.report_dir, named by an io.<key> entry."""
        out_dir = self.cfg["io"]["report_dir"]
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, os.path.basename(self.cfg["io"][key]))

    def raw_input_path(self):
        """The raw acquisitions file, as preprocessing sees it."""
        return self.pre["io"]["input_path"]

    def dataset_path(self, design):
        """flat_dataset.csv / hazard_dataset.csv under the output dir."""
        name = "flat_dataset.csv" if design == "flat" else "hazard_dataset.csv"
        return os.path.join(self.pre["io"]["output_dir"], name)

    def label_column(self, design):
        """The label column differs by design; both names are in schema."""
        return (self.schema["flat_label"] if design == "flat"
                else self.schema["hazard_label"])

    def model_output(self, name):
        return os.path.join(self.mdl["io"]["output_dir"], name)

    def model_artifact_path(self):
        path = self.cfg["io"]["model_artifact"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path


# =====================================================================
# importing the stage scripts as modules
# =====================================================================

_MODULE_CACHE = {}


def import_script(path, alias):
    """
    Import a standalone script by file path under an explicit name.

    preprocess.py and model.py both define load_config and load_data;
    loading them under distinct module names keeps them apart.
    """
    path = resolve(path, ROOT)
    if alias in _MODULE_CACHE:
        return _MODULE_CACHE[alias]
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    _MODULE_CACHE[alias] = module
    return module


def preprocess_module(ctx):
    return import_script(ctx.cfg["io"]["preprocess_script"], "nbo_preprocess")


def model_module(ctx):
    return import_script(ctx.cfg["io"]["model_script"], "nbo_model")


# =====================================================================
# running a stage
# =====================================================================

def run_script(script_path, args, cwd=None, label=None):
    """
    Run a pipeline script and stream its output. Raises on failure -- a
    report built on a half-finished stage is worse than no report.
    """
    script_path = resolve(script_path, ROOT)
    cwd = cwd or os.path.dirname(script_path)
    # The script is named absolutely because cwd is the config's
    # directory, which is not necessarily the script's.
    cmd = [sys.executable, script_path] + [str(a) for a in args]
    print("\n$ cd %s && python %s %s"
          % (cwd, os.path.basename(script_path),
             " ".join(str(a) for a in args)))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError("%s failed with exit code %d"
                           % (label or os.path.basename(script_path),
                              result.returncode))


def run_synthetic(ctx):
    """Write the raw file where the preprocessing config expects it."""
    syn = ctx.cfg["synthetic"]
    target = ctx.raw_input_path()
    os.makedirs(os.path.dirname(target), exist_ok=True)
    run_script(ctx.cfg["io"]["synthetic_script"],
               ["--customers", syn["customers"],
                "--out", os.path.abspath(target),
                "--seed", syn["seed"],
                "--base-rate", syn["base_rate"]],
               label="make_synthetic.py")


def run_preprocess(ctx):
    # cwd is the CONFIG's directory, not the script's. preprocess.py
    # resolves its own io: paths against the working directory, so this
    # is what makes "relative to the config file" true in practice --
    # including for a config that lives outside the repo. When the
    # config sits beside the script, as it ships, the two are the same
    # directory and nothing changes.
    run_script(ctx.cfg["io"]["preprocess_script"],
               ["--config", os.path.abspath(ctx.pre_path)],
               cwd=ctx.pre_dir, label="preprocess.py")


def run_model(ctx):
    run_script(ctx.cfg["io"]["model_script"],
               ["--config", os.path.abspath(ctx.mdl_path)],
               cwd=ctx.mdl_dir, label="model.py")
