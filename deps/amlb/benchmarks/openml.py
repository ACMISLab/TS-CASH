from __future__ import annotations

import logging
import re
from typing import cast

import openml
import pandas as pd
from openml import OpenMLTask, OpenMLDataset

from amlb.utils import Namespace, str_sanitize

log = logging.getLogger(__name__)


def is_openml_benchmark(benchmark: str) -> bool:
    """Check if 'benchmark' is a valid identifier for an openml task or suite."""
    return re.match(r"(openml|test\.openml)/[st]/\d+", benchmark) is not None


def load_oml_benchmark(benchmark: str) -> tuple[str, str | None, list[Namespace]]:
    """Loads benchmark defined by openml suite or task, from openml/s/X or openml/t/Y."""
    domain, oml_type, oml_id_str = benchmark.split("/")
    try:
        oml_id = int(oml_id_str)
    except ValueError:
        raise ValueError(
            f"Could not convert OpenML id {oml_id_str!r} in {benchmark!r} to integer."
        )

    if domain == "test.openml":
        log.debug("Setting openml server to the test server.")
        openml.config.server = "https://test.openml.org/api/v1/xml"

    if openml.config.retry_policy != "robot":
        log.debug(
            "Setting openml retry_policy from '%s' to 'robot'."
            % openml.config.retry_policy
        )
        openml.config.set_retry_policy("robot")

    if oml_type == "t":
        tasks = load_openml_task_as_definition(domain, oml_id)
    elif oml_type == "s":
        tasks = load_openml_tasks_from_suite(domain, oml_id)
    else:
        raise ValueError(f"The oml_type is {oml_type} but must be 's' or 't'")
    # The first argument needs to remain parsable further in the pipeline as is
    # The second argument is path, the benchmark does not exist on disk
    return benchmark, None, tasks


def load_openml_tasks_from_suite(domain: str, oml_id: int) -> list[Namespace]:
    log.info("Loading openml suite %s.", oml_id)
    suite = openml.study.get_suite(oml_id)
    # Here we know the (task, dataset) pairs so only download dataset meta-data is sufficient
    tasks = []
    datasets = cast(
        pd.DataFrame,
        openml.datasets.list_datasets(data_id=suite.data, output_format="dataframe"),
    )
    datasets.set_index("did", inplace=True)
    for tid, did in zip(cast(list[int], suite.tasks), cast(list[int], suite.data)):
        tasks.append(
            Namespace(
                name=str_sanitize(datasets.loc[did]["name"]),
                description=f"{domain}/d/{did}",
                openml_task_id=tid,
                id="{}.org/t/{}".format(domain, tid),
            )
        )
    return tasks


def load_openml_task_as_definition(domain: str, oml_id: int) -> list[Namespace]:
    log.info("Loading openml task %s.", oml_id)
    task, data = load_openml_task_and_data(oml_id)
    return [
        Namespace(
            name=str_sanitize(data.name),
            description=data.description,
            openml_task_id=task.id,
            id="{}.org/t/{}".format(domain, task.id),
        )
    ]


def load_openml_task_and_data(
        task_id: int, with_data: bool = False
) -> tuple[OpenMLTask, OpenMLDataset]:
    task = openml.tasks.get_task(task_id, download_data=False, download_qualities=False)
    data = openml.datasets.get_dataset(
        task.dataset_id, download_data=with_data, download_qualities=False
    )
    return task, data
