import argparse
from dataclasses import replace
import h5py
import logging
import logging.config
import multiprocessing.pool
import os
import random
import shutil
import sys
from typing import List

import docker
import psutil

from .definitions import (Definition, InstantiationStatus, algorithm_status,
                                     get_definitions, list_algorithms)
from .constants import INDEX_DIR
from .datasets import DATASETS, get_dataset
from .results import build_result_filepath
from .runner import run, run_docker


logging.config.fileConfig("logging.conf")
logger = logging.getLogger("annb")


def positive_int(input_str: str) -> int:
    """
    Validates if the input string can be converted to a positive integer.

    Args:
        input_str (str): The input string to validate and convert to a positive integer.

    Returns:
        int: The validated positive integer.

    Raises:
        argparse.ArgumentTypeError: If the input string cannot be converted to a positive integer.
    """
    try:
        i = int(input_str)
        if i < 1:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(f"{input_str} is not a positive integer")

    return i


def run_worker(cpu: int, args: argparse.Namespace, queue: multiprocessing.Queue) -> None:
    """
    Executes the algorithm based on the provided parameters.

    The algorithm is either executed directly or through a Docker container based on the `args.local`
     argument. The function runs until the queue is emptied. When running in a docker container, it 
    executes the algorithm in a Docker container.

    Args:
        cpu (int): The CPU number to be used in the execution.
        args (argparse.Namespace): User provided arguments for running workers. 
        queue (multiprocessing.Queue): The multiprocessing queue that contains the algorithm definitions.

    Returns:
        None
    """
    while not queue.empty():
        definition = queue.get()
        if args.local:
            run(definition, args.dataset, args.count, args.runs, args.batch)
        else:
            memory_margin = 500e6  # reserve some extra memory for misc stuff
            mem_limit = int((psutil.virtual_memory().available - memory_margin) / args.parallelism)
            cpu_limit = str(cpu) if not args.batch else f"0-{multiprocessing.cpu_count() - 1}"
            
            run_docker(definition, args.dataset, args.count, args.runs, args.timeout, args.batch, cpu_limit, mem_limit)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset",
        metavar="NAME",
        help="the dataset to load training points from",
        default="glove-100-angular",
        choices=DATASETS.keys(),
    )
    parser.add_argument(
        "-k", "--count", default=10, type=positive_int, help="the number of near neighbours to search for"
    )
    parser.add_argument(
        "--definitions", metavar="FOLDER", help="base directory of algorithms. Algorithm definitions expected at 'FOLDER/*/config.yml'", default="ann_benchmarks/algorithms"
    )
    parser.add_argument("--algorithm", metavar="NAME", help="run only the named algorithm", default=None)
    parser.add_argument(
        "--docker-tag", metavar="NAME", help="run only algorithms in a particular docker image", default=None
    )
    parser.add_argument(
        "--list-algorithms", help="print the names of all known algorithms and exit", action="store_true"
    )
    parser.add_argument("--force", help="re-run algorithms even if their results already exist", action="store_true")
    parser.add_argument(
        "--runs",
        metavar="COUNT",
        type=positive_int,
        help="run each algorithm instance %(metavar)s times and use only" " the best result",
        default=5,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout (in seconds) for each individual algorithm run, or -1" "if no timeout should be set",
        default=2 * 3600,
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="If set, then will run everything locally (inside the same " "process) rather than using Docker",
    )
    parser.add_argument("--batch", action="store_true", help="If set, algorithms get all queries at once")
    parser.add_argument(
        "--max-n-algorithms", type=int, help="Max number of algorithms to run (just used for testing)", default=-1
    )
    parser.add_argument("--run-disabled", help="run algorithms that are disabled in algos.yml", action="store_true")
    parser.add_argument("--parallelism", type=positive_int, help="Number of Docker containers in parallel", default=1)

    args = parser.parse_args()
    if args.timeout == -1:
        args.timeout = None
    return args


def filter_already_run_definitions(
    definitions: List[Definition], 
    dataset: str, 
    count: int, 
    batch: bool, 
    force: bool
) -> List[Definition]:
    """Filters out the algorithm definitions based on whether they have already been run or not.

    This function checks if there are existing results for each definition by constructing the 
    result filename from the algorithm definition and the provided arguments. If there are no 
    existing results or if the parameter `force=True`, the definition is kept. Otherwise, it is
    discarded.

    Args:
        definitions (List[Definition]): A list of algorithm definitions to be filtered.
        dataset (str): The name of the dataset to load training points from.
        force (bool): If set, re-run algorithms even if their results already exist.

        count (int): The number of near neighbours to search for (only used in file naming convention).
        batch (bool): If set, algorithms get all queries at once (only used in file naming convention).

    Returns:
        List[Definition]: A list of algorithm definitions that either have not been run or are 
                          forced to be re-run.
    """
    filtered_definitions = []

    for definition in definitions:
        not_yet_run = [
            query_args 
            for query_args in (definition.query_argument_groups or [[]])
            if force or not os.path.exists(build_result_filepath(dataset, count, definition, query_args, batch))
        ]

        if not_yet_run:
            definition = replace(definition, query_argument_groups=not_yet_run) if definition.query_argument_groups else definition
            filtered_definitions.append(definition)
            
    return filtered_definitions


def filter_by_available_docker_images(definitions: List[Definition]) -> List[Definition]:
    """
    Filters out the algorithm definitions that do not have an associated, available Docker images.

    This function uses the Docker API to list all Docker images available in the system. It 
    then checks the Docker tags associated with each algorithm definition against the list 
    of available Docker images, filtering out those that are unavailable. 

    Args:
        definitions (List[Definition]): A list of algorithm definitions to be filtered.

    Returns:
        List[Definition]: A list of algorithm definitions that are associated with available Docker images.
    """
    docker_client = docker.from_env()
    docker_tags = {tag.split(":")[0] for image in docker_client.images.list() for tag in image.tags}

    missing_docker_images = set(d.docker_tag for d in definitions).difference(docker_tags)
    if missing_docker_images:
        logger.info(f"not all docker images available, only: {docker_tags}")
        logger.info(f"missing docker images: {missing_docker_images}")
        definitions = [d for d in definitions if d.docker_tag in docker_tags]
    
    return definitions


def check_module_import_and_constructor(df: Definition) -> bool:
    """
    Verifies if the algorithm module can be imported and its constructor exists.

    This function checks if the module specified in the definition can be imported. 
    Additionally, it verifies if the constructor for the algorithm exists within the 
    imported module.

    Args:
        df (Definition): A definition object containing the module and constructor 
        for the algorithm.

    Returns:
        bool: True if the module can be imported and the constructor exists, False 
        otherwise.
    """
    status = algorithm_status(df)
    if status == InstantiationStatus.NO_CONSTRUCTOR:
        raise Exception(
            f"{df.module}.{df.constructor}({df.arguments}): error: the module '{df.module}' does not expose the named constructor"
        )
    if status == InstantiationStatus.NO_MODULE:
        logging.warning(
            f"{df.module}.{df.constructor}({df.arguments}): the module '{df.module}' could not be loaded; skipping"
        )
        return False
    
    return True

def create_workers_and_execute(definitions: List[Definition], args: argparse.Namespace):
    """
    Manages the creation, execution, and termination of worker processes based on provided arguments.

    Args:
        definitions (List[Definition]): List of algorithm definitions to be processed.
        args (argparse.Namespace): User provided arguments for running workers. 

    Raises:
        Exception: If the level of parallelism exceeds the available CPU count or if batch mode is on with more than 
                   one worker.
    """
    cpu_count = multiprocessing.cpu_count()
    if args.parallelism > cpu_count - 1:
        raise Exception(f"Parallelism larger than {cpu_count - 1}! (CPU count minus one)")

    if args.batch and args.parallelism > 1:
        raise Exception(
            f"Batch mode uses all available CPU resources, --parallelism should be set to 1. (Was: {args.parallelism})"
        )

    task_queue = multiprocessing.Queue()
    for definition in definitions:
        task_queue.put(definition)

    try:
        workers = [multiprocessing.Process(target=run_worker, args=(i + 1, args, task_queue)) for i in range(args.parallelism)]
        [worker.start() for worker in workers]
        [worker.join() for worker in workers]
    finally:
        logger.info("Terminating %d workers" % len(workers))
        [worker.terminate() for worker in workers]


def filter_disabled_algorithms(definitions: List[Definition]) -> List[Definition]:
    """
    Excludes disabled algorithms from the given list of definitions.

    This function filters out the algorithm definitions that are marked as disabled in their `config.yml`.

    Args:
        definitions (List[Definition]): A list of algorithm definitions.

    Returns:
        List[Definition]: A list of algorithm definitions excluding any that are disabled.
    """
    disabled_algorithms = [d for d in definitions if d.disabled]
    if disabled_algorithms:
        logger.info(f"Not running disabled algorithms {disabled_algorithms}")

    return [d for d in definitions if not d.disabled]


def limit_algorithms(definitions: List[Definition], limit: int) -> List[Definition]:
    """
    Limits the number of algorithm definitions based on the given limit.

    If the limit is negative, all definitions are returned. For valid 
    sampling, `definitions` should be shuffled before `limit_algorithms`.

    Args:
        definitions (List[Definition]): A list of algorithm definitions.
        limit (int): The maximum number of definitions to return.

    Returns:
        List[Definition]: A trimmed list of algorithm definitions.
    """
    return definitions if limit < 0 else definitions[:limit]


def main():
    args = parse_arguments()

    if args.list_algorithms:
        list_algorithms(args.definitions)
        sys.exit(0)

    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)

    dataset, dimension = get_dataset(args.dataset)
    definitions: List[Definition] = get_definitions(
        dimension=dimension,
        point_type=dataset.attrs.get("point_type", "float"),
        distance_metric=dataset.attrs["distance"],
        count=args.count,
        base_dir=args.definitions,
    )
    random.shuffle(definitions)

    definitions = filter_already_run_definitions(definitions, 
        dataset=args.dataset, 
        count=args.count, 
        batch=args.batch, 
        force=args.force,
    )

    if args.algorithm:
        logger.info(f"running only {args.algorithm}")
        definitions = [d for d in definitions if d.algorithm == args.algorithm]

    if not args.local:
        definitions = filter_by_available_docker_images(definitions)
    else:
        definitions = list(filter(
            check_module_import_and_constructor, definitions
        ))

    definitions = filter_disabled_algorithms(definitions) if not args.run_disabled else definitions
    definitions = limit_algorithms(definitions, args.max_n_algorithms)

    if len(definitions) == 0:
        raise Exception("Nothing to run")
    else:
        logger.info(f"Order: {definitions}")

    create_workers_and_execute(definitions, args)