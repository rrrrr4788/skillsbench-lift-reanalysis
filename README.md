# What Does Skill Lift Actually Measure?

Research artifacts for the source audit and noise-aware reanalysis of SkillsBench by Haozhe Zhang and Zhen Long. This repository is being prepared for public release. The manuscript is a draft.

- [Short paper PDF](paper/main.pdf), [full paper PDF](paper/main_full.pdf), and [Overleaf source ZIP](paper/overleaf-source.zip).
- `data/provenance_tables/` contains the normalized trial and cell tables, model features, and selected audit evidence.
- `reanalysis/results/` contains the frozen statistical outputs, the task tests, the held-out score records, and the reviewed three-task summary.
- `artifacts/` and `docs/section3/` contain controlled-check records, proposed instruction patches, and source evidence. The patches are proposals for future agent experiments.

## Reproduce the statistical analysis

Use Python 3.10 or later. The environment for the frozen outputs is recorded in [the analysis manifest](reanalysis/MANIFEST.json). Run these commands from the repository root.

```sh
python -m pip install -r requirements.txt
python -B reanalysis/check_inputs.py .
python -B reanalysis/skillgate_checks.py . outputs/reproduction base
python -B reanalysis/skillgate_checks.py . outputs/reproduction exact
python -B reanalysis/skillgate_checks.py . outputs/reproduction tasks 100000
python -B reanalysis/skillgate_checks.py . outputs/reproduction seeds 8
python -B reanalysis/skillgate_checks.py . outputs/reproduction null 60
python -B reanalysis/skillgate_checks.py . outputs/reproduction split 60
```

Check that all six commands succeeded and that the output directory contains six JSON files, `task_tests.csv`, and `split_records_forest_policies.csv.gz`. The archived task table contains 83 rows and the forest score table contains 72,120 records. `compare_runs.py` reports differences for files present on both sides. Its exit code alone does not establish complete coverage, and software environments can differ in their last floating-point digits.

To regenerate the figures from the frozen results, run `python -B reanalysis/make_figures.py reanalysis/results outputs/figures`. The generated task figure uses the same data and groups as the paper. The paper version adjusts the axis tick layout. The three-task classification in `reanalysis/results/section3_revision.json` supersedes the historical five-candidate grouping preserved in the frozen inputs and statistical outputs.

## Evidence scope

The statistical analysis starts from the included normalized tables. The controlled-check helper requires the external SkillsBench task files at the revision identified in [the source manifest](data/provenance_tables/A01/source_manifest.json). The catalog and leakage counts rely on upstream review exports. Those exports and raw trajectories are external dependencies.

The [license](LICENSE) covers the project software under its original terms. Upstream materials retain their own terms and attribution. Before changing repository visibility, confirm redistribution permissions for the included derived data and evidence, and replace the manuscript's repository URL placeholder with its final address.
