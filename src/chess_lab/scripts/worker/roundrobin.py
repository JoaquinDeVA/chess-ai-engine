#!/usr/bin/env python3
"""Play several SPRT matches in turns, a fixed number of rounds each, on the worker.

python roundrobin.py queue.json --chunk-rounds 100 --concurrency 4

Runs on the worker (standard library only), from ``~/worker`` where ``uci``, ``fastchess`` and ``openings.epd`` are.
Each turn resumes one match for ``--chunk-rounds`` rounds (2 games per round) and appends a line to
``results/<name>/llr.csv``. A match leaves the queue when fastchess ends its SPRT (H0 or H1 accepted) or when it reaches
``max_rounds`` (no verdict). The queue file is read again before every turn, so jobs can be added while it runs.
``touch results/STOP`` finishes the current turn and exits; SIGTERM does the same at once (fastchess gets SIGINT, which
keeps the state of its SPRT: never kill it with SIGKILL).

Queue file::

    {"jobs": [{"name": "see_off", "a": ["option.Techniques=incremental,see"], "b": ["option.Techniques=incremental"],
               "each": ["tc=10+0.1", "proto=uci", "option.Hash=16"], "elo0": 0, "elo1": 5, "max_rounds": 6000}]}

``a`` and ``b`` are extra fastchess options of each engine (both run ``./uci``); ``each`` defaults to the screening
control. Resuming edits ``rounds`` in the ``config.json`` that fastchess writes on a clean exit.
"""

import argparse
import csv
import json
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_EACH = ["tc=10+0.1", "proto=uci", "option.Hash=16"]
DEFAULT_MAX_ROUNDS = 6000
MAX_FAILURES = 3
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LLR = re.compile(r"LLR: (-?[\d.]+) .*\[(-?[\d.]+), (-?[\d.]+)\]")
_ELO = re.compile(r"(?<!n)Elo: (-?[\d.]+|inf|-inf|nan) \+/- ([\d.]+|inf|nan)")
_GAMES = re.compile(r"Games: (\d+),")
_VERDICT = re.compile(r"completed - (H[01]) was accepted")


def parse_log(text: str) -> Dict[str, Any]:
    """Last figures and verdict in the log of fastchess.

    :param text: Log of the match (it can be the tail only).
    :return: ``games``, ``llr``, ``elo``, ``err`` (None when the log has no block yet) and ``verdict``
        (``"H0"``, ``"H1"`` or None).
    """
    text = _ANSI.sub("", text)
    out: Dict[str, Any] = {"games": None, "llr": None, "elo": None, "err": None, "verdict": None}
    for key, rx in (("games", _GAMES), ("llr", _LLR), ("elo", _ELO)):
        found = rx.findall(text)
        if not found:
            continue
        last = found[-1]
        if key == "games":
            out["games"] = int(last)
        elif key == "llr":
            out["llr"] = float(last[0])
        else:
            out["elo"], out["err"] = (float(v) for v in last)
    verdict = _VERDICT.findall(text)
    out["verdict"] = verdict[-1] if verdict else None
    return out


def rounds_played(config: Dict[str, Any]) -> int:
    """Rounds already played according to the statistics saved by fastchess.

    :param config: Content of ``config.json``.
    :return: Games of the match divided by two.
    """
    stats = next(iter(config.get("stats", {}).values()), {})
    return (stats.get("wins", 0) + stats.get("losses", 0) + stats.get("draws", 0)) // 2


def next_rounds(played: int, chunk: int, max_rounds: int) -> int:
    """Value of ``rounds`` for the coming turn.

    :param played: Rounds played so far.
    :param chunk: Rounds per turn.
    :param max_rounds: Cap of the job.
    :return: ``played + chunk`` without passing the cap.
    """
    return min(played + chunk, max_rounds)


def command(
    job: Dict[str, Any], rounds: int, concurrency: int, config: Path, pgn: Path, resume: bool
) -> List[str]:
    """Command line of fastchess for one turn.

    :param job: Entry of the queue.
    :param rounds: Total rounds to reach at the end of the turn (first turn only).
    :param concurrency: Games at the same time (first turn only).
    :param config: ``config.json`` of the match.
    :param pgn: Output PGN.
    :param resume: True to continue from ``config``.
    :return: Arguments, ``./fastchess`` first.
    """
    if resume:
        return ["./fastchess", "-config", f"file={config}", f"outname={config}"]
    return [
        "./fastchess",
        *["-engine", "cmd=./uci", "name=A", *job.get("a", [])],
        *["-engine", "cmd=./uci", "name=B", *job.get("b", [])],
        *["-each", *job.get("each", DEFAULT_EACH)],
        *["-openings", "file=openings.epd", "format=epd", "order=random"],
        *["-pgnout", f"file={pgn}"],
        *[
            "-sprt",
            f"elo0={job['elo0']}",
            f"elo1={job['elo1']}",
            "alpha=0.05",
            "beta=0.05",
            "model=normalized",
        ],
        *["-rounds", str(rounds), "-games", "2", "-concurrency", str(concurrency)],
        *[
            "-ratinginterval",
            "20",
            "-srand",
            str(job.get("srand", 1)),
            "-config",
            f"outname={config}",
        ],
    ]


def utcnow() -> datetime:
    """Current UTC time (``datetime.UTC`` does not exist in the Python 3.10 of the worker)."""
    return datetime.now(timezone.utc)  # noqa: UP017


class Queue:
    """The jobs, their state files and the turn loop."""

    def __init__(self, queue_file: Path, root: Path, chunk: int, concurrency: int) -> None:
        self.queue_file = queue_file
        self.root = root
        self.chunk = chunk
        self.concurrency = concurrency
        self.child: Optional[subprocess.Popen] = None
        self.stopping = False

    def state_file(self, name: str) -> Path:
        """``state.json`` of a job."""
        return self.root / "results" / name / "state.json"

    def state(self, name: str) -> Dict[str, Any]:
        """Saved state of a job (``running`` when it has none yet)."""
        path = self.state_file(name)
        return (
            json.loads(path.read_text()) if path.exists() else {"status": "running", "failures": 0}
        )

    def save_state(self, name: str, state: Dict[str, Any]) -> None:
        """Write the state of a job."""
        path = self.state_file(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=1))

    def log(self, message: str) -> None:
        """Print a line with the UTC time."""
        line = f"{utcnow():%Y-%m-%d %H:%M:%S} {message}"
        print(line, flush=True)

    def turn(self, job: Dict[str, Any]) -> None:
        """One turn of one job: resume (or start) the match, then record the LLR and decide."""
        name = job["name"]
        directory = self.root / "results" / name
        directory.mkdir(parents=True, exist_ok=True)
        config, pgn, log = (
            directory / "config.json",
            directory / "match.pgn",
            directory / "match.log",
        )
        state = self.state(name)
        max_rounds = job.get("max_rounds", DEFAULT_MAX_ROUNDS)
        resume = config.exists()
        played = rounds_played(json.loads(config.read_text())) if resume else 0
        target = next_rounds(played, self.chunk, max_rounds)
        if resume:
            data = json.loads(config.read_text())
            data["rounds"] = target
            config.write_text(json.dumps(data))
        with log.open("ab") as out:
            self.child = subprocess.Popen(
                command(job, target, self.concurrency, config, pgn, resume),
                cwd=self.root,
                stdout=out,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            code = self.child.wait()
        self.child = None
        tail = log.read_bytes()[-200_000:].decode("utf-8", "replace")
        found = parse_log(tail)
        played_now = rounds_played(json.loads(config.read_text())) if config.exists() else played
        if code != 0 and played_now == played:
            state["failures"] = state.get("failures", 0) + 1
            if state["failures"] >= MAX_FAILURES:
                state["status"] = "error"
        else:
            state["failures"] = 0
        if found["verdict"]:
            state["status"] = found["verdict"]
        elif played_now >= max_rounds:
            state["status"] = "cap"
        state.update(rounds=played_now, llr=found["llr"], elo=found["elo"], err=found["err"])
        self.save_state(name, state)
        csv_path = directory / "llr.csv"
        new = not csv_path.exists()
        with csv_path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            if new:
                writer.writerow(["utc", "games", "llr", "elo", "err", "status"])
            writer.writerow(
                [
                    utcnow().isoformat(timespec="seconds"),
                    2 * played_now,
                    found["llr"],
                    found["elo"],
                    found["err"],
                    state["status"],
                ]
            )
        self.log(
            f"{name}: {2 * played_now} games, LLR {found['llr']}, Elo {found['elo']} +/- {found['err']}, {state['status']}"
        )

    def run(self) -> None:
        """Turns until every job has left the queue or a stop is asked."""
        stop_file = self.root / "results" / "STOP"
        while not self.stopping and not stop_file.exists():
            jobs = json.loads(self.queue_file.read_text())["jobs"]
            pending = [job for job in jobs if self.state(job["name"])["status"] == "running"]
            if not pending:
                self.log("queue finished")
                return
            for job in pending:
                if self.stopping or stop_file.exists():
                    break
                self.turn(job)
        self.log("stopped")

    def interrupt(self, *_: Any) -> None:
        """Signal handler: ask fastchess to stop cleanly and leave the loop."""
        self.stopping = True
        if self.child is not None:
            self.child.send_signal(signal.SIGINT)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point: run the queue until it is empty or a stop is asked."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("queue", type=Path, help="JSON file with the jobs")
    parser.add_argument(
        "--chunk-rounds", type=int, default=100, help="rounds (2 games each) per turn"
    )
    parser.add_argument("--concurrency", type=int, default=4, help="games at the same time")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="folder with uci, fastchess, openings.epd, results/",
    )
    args = parser.parse_args(argv)
    queue = Queue(args.queue.resolve(), args.root.resolve(), args.chunk_rounds, args.concurrency)
    signal.signal(signal.SIGTERM, queue.interrupt)
    signal.signal(signal.SIGINT, queue.interrupt)
    (queue.root / "results").mkdir(exist_ok=True)
    queue.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
