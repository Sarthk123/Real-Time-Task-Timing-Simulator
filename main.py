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
    def __init__(self, task_specs: List[TaskSpec], scheduler: SchedulerBase, sim_time: int = 200, time_unit:int = 1):
        self.specs = task_specs
        self.scheduler = scheduler
        self.sim_time = sim_time
        self.time_unit = time_unit
        self.jobs: List[Job] = []       # all job instances (finished or not)
        self.ready: List[Job] = []
        self.running: Optional[Job] = None
        self.timeline: List[Optional[int]] = [None] * sim_time  # uid of job running at each tick or None
        self.gantt_segments: Dict[str, List[Tuple[int,int]]] = {}  # task_id -> list of (start, end)
        self.job_records: Dict[int, Job] = {}
        self.uid_counter = 1

    def instantiate_job(self, spec: TaskSpec, release_time: int, instance_no: int) -> Job:
        rel = release_time
        # jitter
        if spec.jitter:
            rel += random.randint(-spec.jitter, spec.jitter)
            if rel < 0:
                rel = 0
        rel = int(rel)
        if spec.deadline is not None:
            abs_deadline = rel + spec.deadline
        elif spec.period is not None:
            abs_deadline = rel + spec.period
        else:
            # default: soft deadline = rel + wcet * 10 (arbitrary)
            abs_deadline = rel + spec.wcet * 10
        job = Job(uid=self.uid_counter, task_id=spec.id, release_time=rel, absolute_deadline=abs_deadline,
                  exec_time=spec.wcet, remaining=spec.wcet, spec=spec, instance_no=instance_no)
        self.uid_counter += 1
        self.jobs.append(job)
        self.job_records[job.uid] = job
        return job

    def prepare_job_releases(self):
        """Generate job release times for periodic tasks up to sim_time."""
        releases = []
        for spec in self.specs:
            if spec.type == "periodic" and spec.period is not None:
                # schedule periodic instances
                instances = spec.instances if spec.instances is not None else math.inf
                t = spec.release
                i = 1
                while t < self.sim_time and i <= instances:
                    releases.append((t, spec, i))
                    t += spec.period
                    i += 1
            else:
                # aperiodic/one-shot
                if spec.type in ("aperiodic", "interrupt"):
                    releases.append((spec.release, spec, 1))
        # also include any random arrivals or other dynamic arrivals (not implemented here, but structure allows adding)
        releases.sort(key=lambda x: x[0])
        return releases

    def run(self, realtime_visualize: bool=False, verbose: bool = False):
        # instantiate initial releases structure
        releases = self.prepare_job_releases()
        release_idx = 0
        # For aperiodic tasks whose release >0, they'll be included in releases above

        # main discrete-time simulation
        for t in range(self.sim_time):
            # release new jobs with release_time == t
            while release_idx < len(releases) and releases[release_idx][0] <= t:
                rel_time, spec, inst_no = releases[release_idx]
                job = self.instantiate_job(spec, rel_time, inst_no)
                # if job released later than or equal to current tick, we add to ready at current tick
                if job.release_time <= t:
                    self.ready.append(job)
                else:
                    # schedule for future ticks by reinserting to releases - but our prepare_job_releases ensures release_time set correctly
                    pass
                release_idx += 1

            # select next job
            prev_running = self.running
            candidate = self.scheduler.get_next_job(self.ready, t, self.running)

            # If scheduler returns a job not in ready, ensure it's ready
            if candidate and candidate not in self.ready:
                self.ready.append(candidate)

            # Preemption & switching logic:
            if candidate is None:
                # CPU idle
                self.running = None
                self.timeline[t] = None
            else:
                # if candidate != running -> switch (preemption)
                if self.running is None:
                    # start candidate
                    self.running = candidate
                    if self.running.start_time is None:
                        self.running.start_time = t
                else:
                    if candidate.uid != self.running.uid:
                        # preempt current job only if scheduler is preemptive
                        if self.scheduler.preemptive:
                            # save current
                            # update start/segments
                            self.running = candidate
                            if self.running.start_time is None:
                                self.running.start_time = t
                        else:
                            # scheduler non-preemptive: continue current
                            candidate = self.running

                # execute for 1 time unit
                self.running.remaining -= 1
                self.timeline[t] = self.running.uid

                # RR special: decrement quantum
                if isinstance(self.scheduler, RR):
                    self.scheduler.consume_quantum(1)

                # if job finished
                if self.running.remaining <= 0:
                    self.running.finish_time = t + 1
                    self.running.response_time = (self.running.finish_time - self.running.release_time)
                    # record gantt segment end handled below
                    # remove from ready
                    try:
                        self.ready.remove(self.running)
                    except ValueError:
                        pass
                    self.running = None

            # update ready list: remove jobs whose remaining <=0 (cleanup)
            self.ready = [j for j in self.ready if j.remaining > 0]

        # after simulation, compute gantt segments from timeline
        uid_to_task = {}
        for job in self.jobs:
            uid_to_task[job.uid] = job.task_id
            if job.task_id not in self.gantt_segments:
                self.gantt_segments[job.task_id] = []

        # collapse timeline into segments per task
        current_uid = None
        seg_start = None
        for t, uid in enumerate(self.timeline):
            if uid != current_uid:
                # close previous
                if current_uid is not None:
                    taskid = uid_to_task.get(current_uid, "IDLE")
                    self.gantt_segments.setdefault(taskid, []).append((seg_start, t))
                # start new
                current_uid = uid
                seg_start = t if uid is not None else t
            # continue
        # close final
        if current_uid is not None:
            taskid = uid_to_task.get(current_uid, "IDLE")
            self.gantt_segments.setdefault(taskid, []).append((seg_start, self.sim_time))

        return self._compute_metrics()

    def _compute_metrics(self):
        # CPU utilization = time busy / sim_time
        busy = sum(1 for x in self.timeline if x is not None)
        cpu_util = busy / self.sim_time if self.sim_time > 0 else 0.0
        # response times among finished jobs
        finished = [j for j in self.jobs if j.finish_time is not None]
        response_times = [j.response_time for j in finished if j.response_time is not None]
        avg_response = statistics.mean(response_times) if response_times else None
        # missed deadlines: a finished job misses if finish_time > absolute_deadline; an unfinished job that is past its deadline also counts
        missed = 0
        for j in self.jobs:
            if j.finish_time is not None:
                if j.finish_time > j.absolute_deadline:
                    missed += 1
            else:
                # unfinished and deadline already passed during sim
                if j.absolute_deadline < self.sim_time:
                    missed += 1
        return {
            "cpu_utilization": cpu_util,
            "avg_response_time": avg_response,
            "missed_deadlines": missed,
            "total_jobs": len(self.jobs),
            "finished_jobs": len(finished),
            "busy_time": busy
        }

    def export_csv(self, filename="simulation_results.csv"):
        # export job records + metrics
        metrics = self._compute_metrics()
        now = datetime.datetime.now().isoformat()
        with open(filename, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["sim_time", self.sim_time])
            writer.writerow(["scheduler", self.scheduler.name])
            writer.writerow(["generated_on", now])
            writer.writerow([])
            writer.writerow(["metrics"])
            for k, v in metrics.items():
                writer.writerow([k, v])
            writer.writerow([])
            writer.writerow(["jobs"])
            writer.writerow(["uid", "task_id", "instance_no", "release", "deadline", "wcet", "start", "finish", "response_time", "missed"])
            for j in sorted(self.jobs, key=lambda x: x.uid):
                missed = 1 if (j.finish_time is None or j.finish_time > j.absolute_deadline) and j.absolute_deadline < self.sim_time else 0
                writer.writerow([j.uid, j.task_id, j.instance_no, j.release_time, j.absolute_deadline, j.exec_time, j.start_time, j.finish_time, j.response_time, missed])
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
        for (s, e) in segs:
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

def main():
    args = parse_args()
    if args.load_json:
        specs = load_tasks_from_json(args.load_json)
    elif args.load_csv:
        specs = load_tasks_from_csv(args.load_csv)
    else:
        specs = SAMPLE_TASKS

    scheduler = get_scheduler_by_name(args.scheduler, quantum=args.quantum)
    sim = Simulator(task_specs=specs, scheduler=scheduler, sim_time=args.sim_time)
    print(f"Running simulation for {args.sim_time} ticks with scheduler {scheduler.name} ...")
    metrics = sim.run()
    print("Simulation complete. Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    fname = sim.export_csv(args.export)
    print(f"Results exported to {fname}")

    plot_gantt_and_animate(sim, title="Real-Time Task Timing Visualizer", playback_speed=args.playback_speed)


if __name__ == "__main__":
    main()
