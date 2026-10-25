# Repository cleanup and review

## Scope

Reviewed the supplied code ZIP and used the supplied manuscript and LaTeX figures for documentation. No Git history, datasets, checkpoints, or working experiment environment were supplied. The original folder layout and all Python/MATLAB source files are retained, including archived experiment variants and attribution comments.

## Changes

- Removed the personal W&B entity default in `script/utils/config.py`: `--entity` now defaults to `None`. When tracking is enabled, supply your own team or use W&B's configured default. No key was found or removed.
- Cleaned trailing whitespace, line endings, and final newlines where the parsed Python syntax tree remains identical. Preserved literal strings, import order, duplicate imports, commented experiments, and execution order.
- Replaced the UTF-16 environment-wide requirements entry point with a UTF-8 runtime list. Every retained runtime version pin is unchanged. Preserved the complete historical export separately, including its Windows-only packages, for provenance.
- Added the previously omitted `kmapper` and `torch-geometric-pool` runtime dependencies. Their original versions are unknown and remain unpinned. Separated auxiliary DOS dependencies into `requirements/optional.txt`.
- Added setuptools packaging with namespace discovery so regression and other directories without `__init__.py` remain included. Kept the existing `script` import namespace and added no launcher wrappers or import-time behavior.
- Reworked ignore rules for credentials, caches, environments, and experiment outputs. Removed blanket JSON and setup.py exclusions so legitimate configuration and packaging can be tracked.
- Rewrote the README with the existing logo, exact paper figure exports, data layout, task configuration, module commands, and manuscript citation.
- Added trusted-input security guidance and figure provenance. No license was invented.

## Validation

The cleanup was checked using Python syntax compilation and AST comparison against every original Python source. After allowing only the `--entity` default replacement, every syntax tree must match exactly. Original MATLAB source and logo bytes are preserved. Local documentation links and wheel source coverage were checked. A wheel was built without installing runtime dependencies.

These checks verify source preservation and packaging, not numerical equivalence of a running model. Full imports, dependency resolution, training, inference, and paper-result reproduction were not executed: PyTorch/PyG and the datasets/checkpoints are absent from the review environment. The dependency list is not a validated lockfile, and no vulnerability database audit was performed.

## Existing findings left unchanged

| Finding | Location / implication |
| --- | --- |
| Placeholder data paths | `script/configs/data_spec.py`: configure `DATA_PATH` before running. Main Hydra checkpoints use `DATA_PATH/output/ckpts`; the imported `SAVED_MODEL_PATH` does not control their location. |
| Single-task shipped default | `script/utils/tgf_param.py`: only `node_gs` / `BC` is active. Multi-task experiments require explicit matching task configuration for train and test. |
| Device selection overrides the CLI device | `script/utils/config.py`: the CUDA string does not interpolate the device ID; use `--device_id -1` for CPU. Correct GPU-index support would require a behavioral fix. |
| Fractional split arguments declared as integers | `--test_ratio` and `--val_ratio` use `type=int` while their defaults are `0.15`. Passing a fractional override fails argparse; defaults remain unchanged. |
| Boolean argument limitations | Several flags use `store_true` with `default=True`, and others use `type=bool`. Do not assume passing the string `False` disables them. |
| Raw-data fallback has an unresolved path | `script/utils/data_util.py` passes a literal `{}/input/tokens/raw/` to `TGS_Handler`. Use prepared edge lists/caches until this path is separately corrected and tested. |
| Both dataset indexes read for uncached input | `loader()` expects both token and social CSV indexes, even for a token-only run. |
| Unsafe deserialization for untrusted inputs | Existing `pickle.load` and `torch.load` calls are preserved for compatibility. Only use trusted caches/checkpoints. |
| Unpinned required imports | `kmapper` and `tgp.poolers` are imported by the main pipeline but were absent from the supplied export. Pin their actual experiment versions after validation. |
| Auxiliary module missing | `script/utils/DOS/subroutines/spotlight.py` imports `metrics`; no matching local module is supplied. Do not install an arbitrary similarly named package. This auxiliary script needs its original dependency restored before use. |
| Auxiliary invocation assumptions | Some DOS scripts use top-level `datasets`, `subroutines`, or `util` imports and working-directory-relative paths. They are not advertised as turnkey root-level entry points. |
| Legacy paths / checkpoint portability | Some utilities retain working-directory-relative paths; some checkpoint loads omit `map_location`. Moving files or changing loaders would need separate runtime validation. |
| No supplied license | Maintainers must choose project licensing and review third-party components (including NetworkDOS, manifold utilities, and attributed snippets) before labeling the repository open source. |

## Security review limits

The snapshot was checked for credential-related assignments, provider token patterns, private-key headers, authenticated URLs, long hexadecimal secrets, account identifiers, private filesystem paths, and credential/artifact filenames. No embedded credential was identified. A personal W&B team identifier was found and neutralized. Common Ethereum/dataset names and public upstream attribution URLs were retained.

The archive has no `.git` history. A clean snapshot cannot establish that an existing remote's commits, branches, tags, issues, or releases contain no secrets. No remote repository was modified.
