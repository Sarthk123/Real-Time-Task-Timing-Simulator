#!/usr/bin/env python3
"""
Real-Time Task Timing Visualizer (Simulator)
Single-file Python implementation.

Dependencies:
- Python 3.8+
- numpy
- matplotlib
- pandas (optional, for CSV load)
"""

import json
import csv
import math
import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import statistics
import os
import datetime

# ------------------------------
# Task & Job Models
# ------------------------------

@dataclass
class TaskSpec:
    """Specification for a task type (periodic or aperiodic)."""
    id: str
    wcet: int                      # worst-case execution time (time units)
    period: Optional[int] = None   # if periodic, the period (and implicit deadline)
    deadline: Optional[int] = None # relative deadline (if None and periodic -> equals period)
    release: int = 0               # initial release time
    instances: Optional[int] = None # number of instances for periodic tasks (None => run until sim end)
    color: Optional[str] = None
    type: str = "aperiodic"        # "periodic" or "aperiodic" or "interrupt"
    priority: Optional[int] = None # fixed priority (if used by fixed-priority schedulers)
    jitter: int = 0                # max jitter added to release for modeling jitter

@dataclass
class Job:
    """A specific job instance derived from a TaskSpec."""
    uid: int
    task_id: str
    release_time: int
    absolute_deadline: int
    exec_time: int
    remaining: int
    start_time: Optional[int] = None
    finish_time: Optional[int] = None
    response_time: Optional[int] = None
    spec: TaskSpec = None
    instance_no: int = 0

# ------------------------------
# Scheduler base + implementations
# ------------------------------

class SchedulerBase:
    """Abstract scheduler. Implement get_next_job(ready_jobs, current_time)."""
    preemptive: bool = True

    def __init__(self, name="BASE"):
        self.name = name

    def add_job(self, job: Job):
        pass

    def remove_job(self, job: Job):
        pass

    def get_next_job(self, ready_jobs: List[Job], current_time: int, running_job: Optional[Job]) -> Optional[Job]:
        raise NotImplementedError

class EDF(SchedulerBase):
    def __init__(self):
        super().__init__("EDF")
        self.preemptive = True

    def get_next_job(self, ready_jobs, current_time, running_job):
        if not ready_jobs:
            return None
        # earliest absolute deadline
        return min(ready_jobs, key=lambda j: j.absolute_deadline)

class RM(SchedulerBase):
    def __init__(self):
        super().__init__("RM")
        self.preemptive = True

    def get_next_job(self, ready_jobs, current_time, running_job):
        if not ready_jobs:
            return None
        # priority = smaller period (for tasks with period defined)
        # fallback: use absolute_deadline if no period
        return min(ready_jobs, key=lambda j: (math.inf if j.spec.period is None else j.spec.period, j.absolute_deadline))

class DM(SchedulerBase):
    def __init__(self):
        super().__init__("DM")
        self.preemptive = True

    def get_next_job(self, ready_jobs, current_time, running_job):
        if not ready_jobs:
            return None
        # priority = shorter relative deadline (deadline - release)
        return min(ready_jobs, key=lambda j: ((j.absolute_deadline - j.release_time) if (j.absolute_deadline - j.release_time) > 0 else math.inf, j.absolute_deadline))

class LLF(SchedulerBase):
    def __init__(self):
        super().__init__("LLF")
        self.preemptive = True

    def get_next_job(self, ready_jobs, current_time, running_job):
        if not ready_jobs:
            return None
        # laxity = (deadline - current_time - remaining)
        def laxity(j):
            return (j.absolute_deadline - current_time - j.remaining)
        # job with smallest laxity
        return min(ready_jobs, key=lambda j: (laxity(j), j.absolute_deadline))

class FCFS(SchedulerBase):
    def __init__(self):
        super().__init__("FCFS")
        self.preemptive = False

    def get_next_job(self, ready_jobs, current_time, running_job):
        if not ready_jobs:
            return None
        # first released first served -> min release_time then uid
        return min(ready_jobs, key=lambda j: (j.release_time, j.uid))

class SRTF(SchedulerBase):
    def __init__(self):
        super().__init__("SRTF")
        self.preemptive = True

    def get_next_job(self, ready_jobs, current_time, running_job):
        if not ready_jobs:
            return None
        return min(ready_jobs, key=lambda j: (j.remaining, j.absolute_deadline))

class RR(SchedulerBase):
    def __init__(self, quantum=2):
        super().__init__("RR")
        self.preemptive = True
        self.quantum = quantum
        self._queue: List[Job] = []
        self._time_slice_left = 0
        self._current: Optional[Job] = None

    def get_next_job(self, ready_jobs, current_time, running_job):
        # maintain internal queue in arrival order
        # when new job arrives add to queue
        # if current job exists and time slice left >0 and job still has remaining, continue it
        # otherwise rotate
        # Build a queue: include all ready jobs in arrival order (release_time, uid), but keep internal ordering across calls
        # Sync internal queue with ready_jobs
        ready_set = {j.uid for j in ready_jobs}
        # remove jobs not in ready
        self._queue = [j for j in self._queue if j.uid in ready_set]
        # add any new ones at end in arrival order
        existing_uids = {j.uid for j in self._queue}
        for j in sorted(ready_jobs, key=lambda x: (x.release_time, x.uid)):
            if j.uid not in existing_uids:
                self._queue.append(j)
                existing_uids.add(j.uid)

        # If running_job still has remaining and is same as current internal, continue if slice left.
        if self._current and self._current.remaining > 0 and self._current.uid in existing_uids and self._time_slice_left > 0:
            return self._current

        # else pick next from queue head
        if not self._queue:
            self._current = None
            self._time_slice_left = 0
            return None
        # rotate if current was at head
        if self._current and self._current in self._queue:
            # move current to end if it's finished or slice exhausted
            try:
                idx = self._queue.index(self._current)
            except ValueError:
                idx = None
            if idx == 0:
                # pop head if finished or exhausted
                self._queue.pop(0)
                # if still has remaining, append to end
                if self._current.remaining > 0:
                    self._queue.append(self._current)
        # pick head
        self._current = self._queue[0]
        self._time_slice_left = self.quantum
        return self._current

    def consume_quantum(self, amount=1):
        self._time_slice_left = max(0, self._time_slice_left - amount)


# ------------------------------
# Simulator core
# ------------------------------

class Simulator:
    def __init__(
        self,
        task_specs: List[TaskSpec],
        scheduler: SchedulerBase,
        sim_time: int = 200,
        time_unit: int = 1,
        preemptive: bool = True,
        cores: int = 1
    ):
        self.specs = task_specs
        self.scheduler = scheduler
        self.sim_time = sim_time
        self.time_unit = time_unit
        self.preemptive = preemptive
        self.cores = cores

        # State variables
        self.jobs: List[Job] = []
        self.ready: List[Job] = []
        self.running: List[Optional[Job]] = [None] * cores  # track job per core
        self.timeline: List[List[Optional[int]]] = [
            [None] * cores for _ in range(sim_time)
        ]  # uid per core per tick

        self.gantt_segments: Dict[str, List[Tuple[int, int, int]]] = {}  # task_id -> list of (core, start, end)
        self.job_records: Dict[int, Job] = {}
        self.uid_counter = 1

    def instantiate_job(self, spec: TaskSpec, release_time: int, instance_no: int) -> Job:
        rel = release_time
        if spec.jitter:
            rel += random.randint(-spec.jitter, spec.jitter)
            rel = max(rel, 0)
        rel = int(rel)

        if spec.deadline is not None:
            abs_deadline = rel + spec.deadline
        elif spec.period is not None:
            abs_deadline = rel + spec.period
        else:
            abs_deadline = rel + spec.wcet * 10  # default soft deadline

        job = Job(
            uid=self.uid_counter,
            task_id=spec.id,
            release_time=rel,
            absolute_deadline=abs_deadline,
            exec_time=spec.wcet,
            remaining=spec.wcet,
            spec=spec,
            instance_no=instance_no,
        )
        self.uid_counter += 1
        self.jobs.append(job)
        self.job_records[job.uid] = job
        return job

    def prepare_job_releases(self):
        releases = []
        for spec in self.specs:
            if spec.type == "periodic" and spec.period is not None:
                instances = spec.instances if spec.instances is not None else math.inf
                t = spec.release
                i = 1
                while t < self.sim_time and i <= instances:
                    releases.append((t, spec, i))
                    t += spec.period
                    i += 1
            elif spec.type in ("aperiodic", "interrupt"):
                releases.append((spec.release, spec, 1))
        releases.sort(key=lambda x: x[0])
        return releases

    def run(self, realtime_visualize: bool = False, verbose: bool = False):
        releases = self.prepare_job_releases()
        release_idx = 0

        for t in range(self.sim_time):
            # Release new jobs
            while release_idx < len(releases) and releases[release_idx][0] <= t:
                rel_time, spec, inst_no = releases[release_idx]
                job = self.instantiate_job(spec, rel_time, inst_no)
                if job.release_time <= t:
                    self.ready.append(job)
                release_idx += 1

            # Assign jobs to cores
            for core_id in range(self.cores):
                current = self.running[core_id]
                candidate = self.scheduler.get_next_job(self.ready, t, current)

                # Preemption logic
                if candidate is None:
                    self.running[core_id] = None
                    self.timeline[t][core_id] = None
                    continue

                if current is None:
                    self.running[core_id] = candidate
                    if candidate.start_time is None:
                        candidate.start_time = t
                elif candidate.uid != current.uid:
                    if self.preemptive and self.scheduler.preemptive:
                        self.running[core_id] = candidate
                        if candidate.start_time is None:
                            candidate.start_time = t
                    else:
                        candidate = current

                # Execute 1 tick
                self.running[core_id].remaining -= 1
                self.timeline[t][core_id] = self.running[core_id].uid

                # Round Robin: quantum control
                if isinstance(self.scheduler, RR):
                    self.scheduler.consume_quantum(1)

                # If job finishes
                if self.running[core_id].remaining <= 0:
                    j = self.running[core_id]
                    j.finish_time = t + 1
                    j.response_time = j.finish_time - j.release_time
                    if j in self.ready:
                        self.ready.remove(j)
                    self.running[core_id] = None

            # Cleanup finished
            self.ready = [j for j in self.ready if j.remaining > 0]

        # Build Gantt segments
        uid_to_task = {j.uid: j.task_id for j in self.jobs}
        for job in self.jobs:
            if job.task_id not in self.gantt_segments:
                self.gantt_segments[job.task_id] = []

        for core_id in range(self.cores):
            current_uid = None
            seg_start = 0
            for t in range(self.sim_time):
                uid = self.timeline[t][core_id]
                if uid != current_uid:
                    if current_uid is not None:
                        task_id = uid_to_task.get(current_uid, "IDLE")
                        self.gantt_segments.setdefault(task_id, []).append(
                            (core_id, seg_start, t)
                        )
                    current_uid = uid
                    seg_start = t
            if current_uid is not None:
                task_id = uid_to_task.get(current_uid, "IDLE")
                self.gantt_segments.setdefault(task_id, []).append(
                    (core_id, seg_start, self.sim_time)
                )

        return self._compute_metrics()

    def _compute_metrics(self):
        busy = sum(
            1
            for t in range(self.sim_time)
            for c in range(self.cores)
            if self.timeline[t][c] is not None
        )
        cpu_util = busy / (self.sim_time * self.cores)
        finished = [j for j in self.jobs if j.finish_time is not None]
        response_times = [j.response_time for j in finished if j.response_time]
        avg_response = statistics.mean(response_times) if response_times else None

        missed = sum(
            1
            for j in self.jobs
            if (j.finish_time and j.finish_time > j.absolute_deadline)
            or (j.finish_time is None and j.absolute_deadline < self.sim_time)
        )

        return {
            "cpu_utilization": cpu_util,
            "avg_response_time": avg_response,
            "missed_deadlines": missed,
            "total_jobs": len(self.jobs),
            "finished_jobs": len(finished),
            "busy_time": busy,
        }

    def export_csv(self, filename="simulation_results.csv"):
        metrics = self._compute_metrics()
        now = datetime.datetime.now().isoformat()
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sim_time", self.sim_time])
            writer.writerow(["scheduler", self.scheduler.name])
            writer.writerow(["cores", self.cores])
            writer.writerow(["preemptive", self.preemptive])
            writer.writerow(["generated_on", now])
            writer.writerow([])
            writer.writerow(["metrics"])
            for k, v in metrics.items():
                writer.writerow([k, v])
            writer.writerow([])
            writer.writerow(["jobs"])
            writer.writerow(
                [
                    "uid",
                    "task_id",
                    "instance_no",
                    "release",
                    "deadline",
                    "wcet",
                    "start",
                    "finish",
                    "response_time",
                    "missed",
                ]
            )
            for j in sorted(self.jobs, key=lambda x: x.uid):
                missed = int(
                    (j.finish_time and j.finish_time > j.absolute_deadline)
                    or (j.finish_time is None and j.absolute_deadline < self.sim_time)
                )
                writer.writerow(
                    [
                        j.uid,
                        j.task_id,
                        j.instance_no,
                        j.release_time,
                        j.absolute_deadline,
                        j.exec_time,
                        j.start_time,
                        j.finish_time,
                        j.response_time,
                        missed,
                    ]
                )
        return filename

# ------------------------------
# Visualization (Matplotlib Gantt)
# ------------------------------

def plot_gantt_and_animate(sim: Simulator, title: str = "RT Scheduling Gantt", playback_speed: float = 1.0):
    # Build list of tasks order (consistent display)
    task_ids = sorted(list(sim.gantt_segments.keys()))
    if "IDLE" in task_ids:
        task_ids.remove("IDLE")
        task_ids = ["IDLE"] + task_ids

    # map tasks to row index
    task_indices = {tid: idx for idx, tid in enumerate(task_ids)}

    fig, ax = plt.subplots(figsize=(12, 1 + 0.5 * len(task_ids)))
    ax.set_title(title + f"  (Scheduler: {sim.scheduler.name})")
    yticks = []
    ylabels = []
    colors = plt.get_cmap('tab20').colors
    color_map = {}
    for i, tid in enumerate(task_ids):
        yticks.append(i + 0.5)
        ylabels.append(tid)
        color_map[tid] = colors[i % len(colors)]

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Time (ticks)")
    ax.set_xlim(0, sim.sim_time)
    ax.set_ylim(0, len(task_ids))
    ax.grid(True, axis='x', linestyle='--', alpha=0.4)

    # Prepare rectangles
    bars = []
    for tid, segs in sim.gantt_segments.items():
        row = task_indices.get(tid, None)
        if row is None:
            continue
        for (_, s, e) in segs: 

            print("DEBUG segs =", segs)
            rect = ax.barh(row + 0.1, e - s, left=s, height=0.8, align='center', color=color_map.get(tid, 'grey'), edgecolor='black')
            bars.append(rect)

    # draw deadlines: for each job, vertical line
    for job in sim.jobs:
        # draw only if deadline within window
        if 0 <= job.absolute_deadline <= sim.sim_time:
            ax.axvline(job.absolute_deadline, color='red', linestyle='--', linewidth=0.6, alpha=0.6)

    # annotate missed deadlines
    # find finished jobs with finish > deadline
    for job in sim.jobs:
        if job.finish_time and job.finish_time > job.absolute_deadline:
            # place red X at finish time at corresponding task row
            tid = job.task_id
            row = task_indices.get(tid, None)
            if row is not None:
                ax.text(job.finish_time + 0.1, row + 0.5, "✖", color='red', fontsize=10)

    # show current time cursor via vertical line in animation
    cursor = ax.axvline(0, color='blue', linewidth=1.2)

    # subplot for metrics
    metrics_text = ax.text(0.01, -0.08, "", transform=ax.transAxes, fontsize=9, va="top", ha="left")

    # animation update function
    def update(frame):
        cursor.set_xdata(frame)
        metrics = sim._compute_metrics()
        mt = f"Time: {frame}/{sim.sim_time}   CPU util (total): {metrics['cpu_utilization']:.3f}   " \
             f"Missed: {metrics['missed_deadlines']}   Finished: {metrics['finished_jobs']}/{metrics['total_jobs']}"
        metrics_text.set_text(mt)
        return cursor, metrics_text

    ani = FuncAnimation(fig, update, frames=range(0, sim.sim_time + 1), interval=max(1, int(500 / playback_speed)), blit=False, repeat=False)
    plt.tight_layout()
    plt.show()


# ------------------------------
# Helpers for loading tasks
# ------------------------------

def load_tasks_from_json(path: str) -> List[TaskSpec]:
    with open(path, "r") as f:
        data = json.load(f)
    specs = []
    for item in data:
        specs.append(TaskSpec(
            id=item.get("id"),
            wcet=int(item.get("wcet")),
            period=(None if item.get("period") is None else int(item.get("period"))),
            deadline=(None if item.get("deadline") is None else int(item.get("deadline"))),
            release=int(item.get("release", 0)),
            instances=(None if item.get("instances") is None else int(item.get("instances"))),
            color=item.get("color"),
            type=item.get("type", "aperiodic"),
            priority=item.get("priority"),
            jitter=int(item.get("jitter", 0))
        ))
    return specs

def load_tasks_from_csv(path: str) -> List[TaskSpec]:
    specs = []
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            specs.append(TaskSpec(
                id=row.get("id"),
                wcet=int(row.get("wcet")),
                period=(None if not row.get("period") else int(row.get("period"))),
                deadline=(None if not row.get("deadline") else int(row.get("deadline"))),
                release=int(row.get("release", 0)),
                instances=(None if not row.get("instances") else int(row.get("instances"))),
                color=row.get("color"),
                type=row.get("type", "aperiodic"),
                priority=(None if not row.get("priority") else int(row.get("priority"))),
                jitter=(0 if not row.get("jitter") else int(row.get("jitter")))
            ))
    return specs

# ------------------------------
# Sample tasks for quick test
# ------------------------------

SAMPLE_TASKS = [
    # periodic tasks: id, wcet, period, (deadline default=period)
    TaskSpec(id="T1", wcet=1, period=4, release=0, type="periodic", instances=20),
    TaskSpec(id="T2", wcet=2, period=6, release=0, type="periodic", instances=15),
    TaskSpec(id="T3", wcet=1, period=8, release=0, type="periodic", instances=10),

    # aperiodic
    TaskSpec(id="A1", wcet=3, release=5, deadline=12, type="aperiodic"),
    TaskSpec(id="A2", wcet=2, release=12, deadline=20, type="aperiodic"),

    # interrupt-style (one-shot arriving later)
    TaskSpec(id="IRQ", wcet=1, release=18, deadline=20, type="interrupt"),
]

# ------------------------------
# CLI / Runner
# ------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Real-Time Task Timing Visualizer (Simulator)")
    p.add_argument("--sim-time", type=int, default=80, help="Simulation length in time units")
    p.add_argument("--scheduler", type=str, default="EDF", choices=["EDF","RM","DM","LLF","FCFS","SRTF","RR"], help="Scheduler to use")
    p.add_argument("--quantum", type=int, default=2, help="Quantum for RR (only used if scheduler=RR)")
    p.add_argument("--load-json", type=str, default=None, help="Load tasks from JSON file")
    p.add_argument("--load-csv", type=str, default=None, help="Load tasks from CSV file")
    p.add_argument("--playback-speed", type=float, default=1.0, help="Animation playback speed (higher -> faster)")
    p.add_argument("--export", type=str, default="simulation_results.csv", help="CSV export filename")
    return p.parse_args()

def get_scheduler_by_name(name: str, quantum: int = 2):
    name = name.upper()
    if name == "EDF":
        return EDF()
    if name == "RM":
        return RM()
    if name == "DM":
        return DM()
    if name == "LLF":
        return LLF()
    if name == "FCFS":
        return FCFS()
    if name == "SRTF":
        return SRTF()
    if name == "RR":
        return RR(quantum=quantum)
    raise ValueError("Unknown scheduler")

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys

# === Your existing imports ===
# from scheduler import get_scheduler_by_name
# from simulator import Simulator
# from visualization import plot_gantt_and_animate
# from data_loader import load_tasks_from_json, load_tasks_from_csv, SAMPLE_TASKS
# (Make sure those imports exist)

def gui_config_menu():
    root = tk.Tk()
    root.title("Scheduler Configuration")
    root.geometry("420x400")

    # Variables
    scheduler_var = tk.StringVar(value="EDF")
    preemption_var = tk.BooleanVar(value=True)
    cores_var = tk.IntVar(value=1)
    sim_time_var = tk.IntVar(value=50)
    quantum_var = tk.IntVar(value=2)
    file_path_var = tk.StringVar(value="")
    file_type_var = tk.StringVar(value="None")

    # === Labels & Inputs ===
    ttk.Label(root, text="Scheduler Type:", font=("Arial", 11)).pack(pady=5)
    sched_combo = ttk.Combobox(root, textvariable=scheduler_var, values=["EDF", "RM", "DM", "LLF"], state="readonly")
    sched_combo.pack()

    ttk.Label(root, text="Preemption:", font=("Arial", 11)).pack(pady=5)
    ttk.Checkbutton(root, text="Enable Preemption", variable=preemption_var).pack()

    ttk.Label(root, text="Number of Cores:", font=("Arial", 11)).pack(pady=5)
    ttk.Spinbox(root, from_=1, to=8, textvariable=cores_var, width=5).pack()

    ttk.Label(root, text="Simulation Time (ticks):", font=("Arial", 11)).pack(pady=5)
    ttk.Entry(root, textvariable=sim_time_var, width=10).pack()

    ttk.Label(root, text="Quantum (for RR or hybrid):", font=("Arial", 11)).pack(pady=5)
    ttk.Entry(root, textvariable=quantum_var, width=10).pack()

    ttk.Label(root, text="Load Task File:", font=("Arial", 11)).pack(pady=5)
    file_frame = ttk.Frame(root)
    file_frame.pack()

    def browse_file():
        filepath = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv")])
        if filepath:
            file_path_var.set(filepath)
            if filepath.endswith(".json"):
                file_type_var.set("json")
            elif filepath.endswith(".csv"):
                file_type_var.set("csv")

    ttk.Button(file_frame, text="Browse...", command=browse_file).pack(side=tk.LEFT, padx=5)
    ttk.Label(file_frame, textvariable=file_path_var, wraplength=300).pack(side=tk.LEFT)

    def submit():
        if sim_time_var.get() <= 0:
            messagebox.showerror("Error", "Simulation time must be positive")
            return
        root.destroy()

    ttk.Button(root, text="Run Simulation", command=submit).pack(pady=20)

    root.mainloop()

    return {
        "scheduler": scheduler_var.get(),
        "preemption": preemption_var.get(),
        "cores": cores_var.get(),
        "sim_time": sim_time_var.get(),
        "quantum": quantum_var.get(),
        "file_path": file_path_var.get(),
        "file_type": file_type_var.get(),
    }


def main():
    # === Get config from GUI ===
    config = gui_config_menu()

    # === Load tasks ===
    if config["file_type"] == "json":
        specs = load_tasks_from_json(config["file_path"])
    elif config["file_type"] == "csv":
        specs = load_tasks_from_csv(config["file_path"])
    else:
        specs = SAMPLE_TASKS

    scheduler = get_scheduler_by_name(config["scheduler"], quantum=config["quantum"])
    sim = Simulator(
        task_specs=specs,
        scheduler=scheduler,
        sim_time=config["sim_time"],
        preemptive=config["preemption"],
        cores=config["cores"]
    )

    print(f"Running simulation for {config['sim_time']} ticks with scheduler {scheduler.name} ...")
    metrics = sim.run()
    print("Simulation complete. Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    fname = sim.export_csv("simulation_output.csv")
    print(f"Results exported to {fname}")

    plot_gantt_and_animate(sim, title=f"{scheduler.name} Visualization", playback_speed=1.0)


if __name__ == "__main__":
    main()
