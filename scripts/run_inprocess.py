"""Run KFFL in a single process, without Ray.

The ServerApp's ``main`` and the ClientApp's handlers are exactly the ones
``flwr run`` executes.  Only the transport is replaced: instead of Flower's
Grid shipping messages to simulated SuperNodes, ``InProcessGrid`` calls each
client handler directly.  Message passing is most of a simulated round's cost
for small models, so this is several times faster and many runs can share a
machine.  Use ``flwr run`` for real deployments and to cross-check.

    PYTHONPATH=. python scripts/run_inprocess.py \\
        --run-config 'dataset="compas" sensitive-attribute="african-american" fairness-lambda=10.0'
"""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

import torch
from flwr.app import Context, Message, RecordDict

APP = Path(__file__).resolve().parents[1]


def load_run_config(overrides: str) -> dict:
    """pyproject defaults, updated by ``key=value`` pairs as given to ``flwr run``."""
    project = tomllib.loads((APP / "pyproject.toml").read_text())
    config = dict(project["tool"]["flwr"]["app"]["config"])
    pairs = re.findall(r'[\w-]+\s*=\s*(?:"[^"]*"|\S+)', overrides)
    config.update(tomllib.loads("\n".join(pairs)))
    return config


class InProcessGrid:
    """The two Grid methods the ServerApp uses, served by local function calls."""

    def __init__(self, run_config: dict, num_clients: int) -> None:
        from kffl import client_app

        self.handlers = {
            "query.fair1": client_app.fair1,
            "query.fair2": client_app.fair2,
            "train.local_update": client_app.local_update,
        }
        self.contexts = {
            node_id: Context(
                run_id=0,
                node_id=node_id,
                node_config={"partition-id": node_id - 1, "num-partitions": num_clients},
                state=RecordDict(),
                run_config=run_config,
            )
            for node_id in range(1, num_clients + 1)
        }

    def get_node_ids(self) -> list[int]:
        return list(self.contexts)

    def send_and_receive(self, messages: list[Message]) -> list[Message]:
        return [
            self.handlers[m.metadata.message_type](m, self.contexts[m.metadata.dst_node_id])
            for m in messages
        ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-config", default="")
    p.add_argument("--num-clients", type=int, default=4)  # m = 4 (App. B.1)
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    from kffl import server_app

    run_config = load_run_config(args.run_config)
    server_context = Context(
        run_id=0, node_id=0, node_config={}, state=RecordDict(), run_config=run_config
    )
    server_app.main(InProcessGrid(run_config, args.num_clients), server_context)


if __name__ == "__main__":
    main()
