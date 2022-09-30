import logging
import gzip
from typing import Tuple, Optional
from qiskit.circuit import qpy_serialization
from qiskit.circuit.library import QuantumVolume
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes.routing.sabre_swap import SabreSwap
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.passes.layout.apply_layout import ApplyLayout
from qiskit.transpiler.passes.layout.sabre_layout import SabreLayout
from dsm_swap.optim import rh_knitter, linear_decay
from dsm_swap.utils import dag_decompose
import os.path
import time
from datetime import datetime
import json
from uuid import uuid4
import argparse
import sys
from functools import partial


_TASK_ID_AUTO = "*auto*"
_QV_SEED_ADD_STEP = 1267


logger = logging.getLogger(__name__)


def dsm_swap_v4(
    qc,
    *,
    swap_layers=None,
    horizon=2,
    restarts=5,
    seed=38492314298,
    max_optim_steps=30,
    layout_pass=True,
    cmap=None,
) -> QuantumCircuit:
    """
    Reference algorithm for testing.
    """
    cmap = CouplingMap.from_line(qc.num_qubits, bidirectional=True) if cmap is None else cmap
    if cmap.size() != qc.num_qubits:
        raise ValueError("Incompatibility between the coupling map and the quantum circuit")
    dag = circuit_to_dag(qc)
    qc = None

    if layout_pass:
        layout = SabreLayout(cmap, max_iterations=3, seed=seed)
        layout.run(dag)
        apply_layout = ApplyLayout()
        apply_layout.property_set["layout"] = layout.property_set["layout"]
        dag = apply_layout.run(dag)

    dagd = dag_decompose(dag)
    layer_perms = rh_knitter(
        dagd.get_adjs(mode="coo"),
        max_optim_steps=max_optim_steps,
        seed=seed,
        horizon=horizon,
        restart=restarts,
        cmap=cmap,
        hw_cost_decay=linear_decay,
        swap_layers=swap_layers,
    )

    return dag_to_circuit(dagd.compose(layer_perms)[0])


def sabre(qc, *, seed=38492314298, layout_pass=True, cmap=None, heuristic="lookahead"):
    cmap = CouplingMap.from_line(qc.num_qubits, bidirectional=True) if cmap is None else cmap
    dag = circuit_to_dag(qc)
    qc = None

    if layout_pass:
        layout = SabreLayout(cmap, max_iterations=3, seed=seed)
        layout.run(dag)
        apply_layout = ApplyLayout()
        apply_layout.property_set["layout"] = layout.property_set["layout"]
        dag = apply_layout.run(dag)

    dag = SabreSwap(cmap, heuristic=heuristic, seed=seed).run(dag)
    return dag_to_circuit(dag)


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-set-seed", type=int, default=344298, help="Seed for the test set generator"
    )
    parser.add_argument("--test-set", type=str, required=True, help="Test set")
    parser.add_argument("--start-offset", type=int, default=0, help="Test set start offset")
    parser.add_argument(
        "--max-jobs", type=int, default=-1, help="Max number of jobs to be executed"
    )

    parser.add_argument("--task-id", type=str, default=_TASK_ID_AUTO, help="Task id")
    parser.add_argument("--user-recs", type=str, default="{}", help="Json-encoded user columns")
    parser.add_argument("--folder", type=str, required=True, help="Destination folder")
    parser.add_argument("--dummy-run", type=bool, default=False, help="Enable dummy run")

    parser.add_argument("--algo", type=str, default="dsm-swap-v4", help="Swap mapper algorithm")
    parser.add_argument(
        "--seed", type=int, default=38492314298, help="Seed for the swap mapper algorithm"
    )
    parser.add_argument(
        "--swap-args", type=str, default="{}", help="Json-encoded swap mapper arguments"
    )
    parser.add_argument("--cmap", type=str, default="line", help="Coupling map")
    parser.add_argument("--save-circ", type=bool, default=False, help="When true save the circuit")

    args = parser.parse_args()
    args.user_recs = json.loads(args.user_recs)
    args.swap_args = json.loads(args.swap_args)
    args.task_id = str(uuid4()) if args.task_id == _TASK_ID_AUTO else args.task_id
    return args


def _get_algo(name):
    if name == "dsm-swap-v4":
        return dsm_swap_v4
    if name == "sabre":
        return sabre
    raise ValueError("Unrecognized algorithm")


def _get_cmap_factory(name):
    if name == "line":
        return partial(CouplingMap.from_line, bidirectional=True)
    if name == "ring":
        return partial(CouplingMap.from_ring, bidirectional=True)
    raise ValueError(f"Unrecognized cmap: {name}")


class QVIterator(object):
    def __init__(self, n, seed, offset=0):
        self.n = n
        self.seed = seed + offset * _QV_SEED_ADD_STEP

    def __iter__(self):
        return self

    def __next__(self):
        qc = QuantumVolume(self.n, seed=self.seed).decompose()
        self.seed += _QV_SEED_ADD_STEP
        return qc


def _get_test_set_iterator(name, *, offset=0, seed=None):
    if name == "nannicini-qv8":
        with gzip.open("../stats/nannicini-qv8/qv256_batch1_original.qpy.gz", "rb") as f:
            circuits_orig1 = qpy_serialization.load(f)
        return circuits_orig1[offset:]
    if name.startswith("qv"):
        n = int(name[2:])
        return QVIterator(n, seed=seed, offset=offset)
    if name.startswith("mcx"):
        n = int(name[3:])
        qc = QuantumCircuit(n)
        qc.mcx(list(range(qc.num_qubits - 1)), qc.num_qubits - 1)
        qc = qc.decompose().decompose()
        return [qc]
    raise ValueError(f"Unrecognized test set: {name}")


def _circ_features(qc, *, postfix="", seed=None):
    qc = transpile(qc, basis_gates=["u3", "cx"], optimization_level=3, seed_transpiler=seed)
    stats = {
        f"circ{postfix}_depth": qc.depth(),
        f"circ{postfix}_size": qc.size(),
        f"circ{postfix}_cnots": qc.count_ops().get("cx", 0),
    }

    dagd = dag_decompose(circuit_to_dag(qc))
    stats.update(
        {
            f"circ{postfix}_layerd": dagd.avg_layer_density(),
            f"circ{postfix}_qubits": dagd.num_qubits(),
            f"circ{postfix}_layers": dagd.size,
        }
    )
    return stats


def main() -> int:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO, stream=sys.stdout
    )
    args = _parse_args()
    cmd_run_id = str(uuid4())
    logger.info(f"Starting task {args.task_id} (cmd_run_id: {cmd_run_id})")
    logger.info(f"Cmd args: {args}")
    if args.dummy_run:
        logger.info(f"Task {args.task_id} completed dummy run (cmd_run_id: {cmd_run_id})")
        return 0

    swap_mapper = _get_algo(args.algo)
    cmap_factory = _get_cmap_factory(args.cmap)
    test_set_it = _get_test_set_iterator(
        args.test_set, offset=args.start_offset, seed=args.test_set_seed
    )
    for idx_, qc in enumerate(test_set_it):
        idx = idx_ + args.start_offset
        prefix = f"{idx:06d}-{str(uuid4())}"
        logger.info(f"Job file prefix {prefix}")

        cmap = cmap_factory(qc.num_qubits)
        start = time.monotonic()
        success = False

        # Obtain stats before swap
        stats0 = _circ_features(qc, postfix="0", seed=args.seed)

        try:
            qc = swap_mapper(qc, seed=args.seed, cmap=cmap, **args.swap_args)
            success = True
        except:
            qc = None
            logger.info("Swap mapper algorithm failure")
        time_taken = time.monotonic() - start

        stats = {
            "srcid": idx,
            "prefix": prefix,
            "start_time": str(datetime.now()),
            "algo": args.algo,
            "task_id": args.task_id,
            "cmap": args.cmap,
            "test_set_seed": args.test_set_seed,
            "test_set": args.test_set,
            "swap_mapper_kwargs": json.dumps(args.swap_args),
            "cmd_run_id": cmd_run_id,
            "time": time_taken,
            "success": success,
            "seed": args.seed,
        }
        stats.update(stats0)
        stats.update(args.user_recs)
        if qc is not None:
            stats.update(_circ_features(qc, seed=args.seed))
        logger.info(f"Record: {json.dumps(stats)}")
        with open(os.path.join(args.folder, f"{prefix}.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)

        if qc is not None and args.save_circ:
            with open(os.path.join(args.folder, f"{prefix}.qpy"), "wb") as fd:
                qpy_serialization.dump(qc, fd)

        if args.max_jobs > 0 and idx_ >= (args.max_jobs - 1):
            logger.info(f"Max jobs reached for task {args.task_id}")
            break

    logger.info(f"Task {args.task_id} completed (cmd_run_id: {cmd_run_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
