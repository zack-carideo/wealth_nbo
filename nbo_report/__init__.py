"""
Reporting layer for the cross-sell pipeline.

Sits on top of the two existing stages and adds nothing to them:

    nbo_data_preprocessing/preprocess.py   raw rows  -> modelling datasets
    nbo_data_modeling/model.py             datasets  -> metrics + scores
    nbo_report/                            everything -> HTML for a human

The two entry points are eda.py (pre-model: what is in the data) and
main.py (post-model: what the model found and who to call), both at the
repo root. They share every function in here, so the sequence profile in
the model report is computed exactly the same way as the one in the EDA
report — only the population differs.

No module in this package contains a column name, a date or a threshold.
All of that comes from report_config.yaml plus the two stage configs.
"""

__all__ = ["config", "charts", "html", "sequences", "profiling", "targeting"]
