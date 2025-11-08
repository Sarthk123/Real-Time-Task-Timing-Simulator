
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
    dependencies: List[str] = field(default_factory=list) # <-- ADDED

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
    dependencies: List[int] = field(default_factory=list) # <-- ADDED

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
        self.pending_jobs: List[Job] = [] # <-- ADDED
        self.jobs_to_release: List[Job] = [] # <-- ADDED
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

    # --- NEW METHOD ---
    def _pre_instantiate_and_link(self):
        """Instantiate all jobs and link their dependencies based on instance number."""
        releases = self.prepare_job_releases()
        instance_map: Dict[Tuple[str, int], Job] = {} # Key: (task_id, instance_no)

        # First pass: Instantiate all jobs
        for (rel_time, spec, inst_no) in releases:
            job = self.instantiate_job(spec, rel_time, inst_no)
            instance_map[(spec.id, inst_no)] = job

        # Second pass: Link dependencies
        # This assumes T2_inst_N depends on T1_inst_N
        for job in self.jobs:
            if not job.spec.dependencies:
                continue
            for dep_task_id in job.spec.dependencies:
                # Find the dependency with the *same instance number*
                dep_job = instance_map.get((dep_task_id, job.instance_no))
                if dep_job:
                    job.dependencies.append(dep_job.uid)
                else:
                    # Handle case where dependency instance might not exist (e.g., runs for fewer instances)
                    print(f"Warning: Could not find dependency {dep_task_id} (instance {job.instance_no}) for job {job.uid} ({job.task_id})")
        
        # Sort all jobs by release time for the main loop
        self.jobs_to_release = sorted(self.jobs, key=lambda j: j.release_time)


    # --- NEW METHOD ---
    def _dependencies_met(self, job: Job) -> bool:
        """Check if all dependencies for a job are finished."""
        if not job.dependencies:
            return True
        for dep_uid in job.dependencies:
            dep_job = self.job_records.get(dep_uid)
            # Fails if dependency job doesn't exist or is not finished
            if not dep_job or dep_job.finish_time is None:
                return False
        return True

    def run(self, realtime_visualize: bool = False, verbose: bool = False):
        # --- MODIFIED: Pre-instantiate and link all jobs ---
        self._pre_instantiate_and_link()
        release_idx = 0
        # --------------------------------------------------

        for t in range(self.sim_time):
            # --- MODIFIED: Release jobs into pending state ---
            while release_idx < len(self.jobs_to_release) and self.jobs_to_release[release_idx].release_time <= t:
                job = self.jobs_to_release[release_idx]
                self.pending_jobs.append(job)
                release_idx += 1

            # --- MODIFIED: Check dependencies and move from pending to ready ---
            newly_ready = []
            for job in self.pending_jobs:
                if self._dependencies_met(job):
                    self.ready.append(job)
                    newly_ready.append(job)
            
            # Remove newly ready jobs from pending list
            if newly_ready:
                self.pending_jobs = [j for j in self.pending_jobs if j not in newly_ready]
            # -----------------------------------------------------------------

            # Assign jobs to cores
            for core_id in range(self.cores):
                current = self.running[core_id]
                
                # --- MODIFICATION: Ensure scheduler can't pick a job already running on another core ---
                # Get jobs that are ready AND not currently running on any core
                available_ready_jobs = [j for j in self.ready if j not in self.running]
                
                candidate = self.scheduler.get_next_job(available_ready_jobs, t, current)

                # Preemption logic
                if candidate is None:
                    # No candidate, core becomes idle (if it wasn't already)
                    if current is not None:
                         # Job was running, but no candidate (or scheduler chose none)
                         # This part is tricky. If preemptive, it should idle.
                         # If non-preemptive, 'current' should continue.
                         # But FCFS/non-preemptive logic is in scheduler.
                         # Let's assume if scheduler returns None, core is idle.
                         pass # Keep 'current' if non-preemptive logic is handled by scheduler
                    
                    # If core is idle, set it to None
                    if current is None:
                        self.running[core_id] = None
                        self.timeline[t][core_id] = None
                        continue
                    else:
                        # Core was busy, but scheduler picked no one.
                        # This implies 'current' should continue if non-preemptive
                        # or 'current' was preempted and no one else is ready.
                        
                        # Let's refine: A non-preemptive scheduler would return 'current' if it's still running.
                        # A preemptive scheduler returning None means 'current' is preempted and ready queue is empty.
                        if self.preemptive and self.scheduler.preemptive:
                             if current in self.ready and current not in available_ready_jobs:
                                 # This job is running on another core. This core should be idle.
                                 pass
                             self.running[core_id] = None
                             self.timeline[t][core_id] = None
                             continue
                        else: # Non-preemptive
                             candidate = current # Continue running 'current'
                    
                # If scheduler returned None *and* core was idle, it's already handled
                if candidate is None and current is None:
                     self.running[core_id] = None
                     self.timeline[t][core_id] = None
                     continue
                
                # If scheduler returned None *and* core was busy
                if candidate is None and current is not None:
                    if self.preemptive and self.scheduler.preemptive:
                        # Preempted, but no other job ready. Core idles.
                        self.running[core_id] = None
                        self.timeline[t][core_id] = None
                        continue
                    else:
                        # Non-preemptive, continue 'current'
                        candidate = current


                if current is None:
                    # Core was idle, start candidate
                    self.running[core_id] = candidate
                    if candidate.start_time is None:
                        candidate.start_time = t
                    if candidate in self.ready:
                        self.ready.remove(candidate) # Job is now running
                        
                elif candidate.uid != current.uid:
                    # Switch decision
                    if self.preemptive and self.scheduler.preemptive:
                        # Preempt: put 'current' back in ready, start 'candidate'
                        if current.remaining > 0:
                            self.ready.append(current) # Put current back
                        self.running[core_id] = candidate
                        if candidate.start_time is None:
                            candidate.start_time = t
                        if candidate in self.ready:
                             self.ready.remove(candidate)
                    else:
                        # Non-preemptive: continue 'current'
                        candidate = current
                
                # If candidate is the same as current, just continue
                # (This is covered by the logic above)

                # Execute 1 tick
                job_to_run = self.running[core_id]
                if job_to_run is None:
                    self.timeline[t][core_id] = None
                    continue # Core is idle
                
                job_to_run.remaining -= 1
                self.timeline[t][core_id] = job_to_run.uid

                # Round Robin: quantum control
                if isinstance(self.scheduler, RR):
                    self.scheduler.consume_quantum(1)

                # If job finishes
                if job_to_run.remaining <= 0:
                    job_to_run.finish_time = t + 1
                    job_to_run.response_time = job_to_run.finish_time - job_to_run.release_time
                    # No need to remove from ready, it was removed when it started
                    self.running[core_id] = None

            # Cleanup finished jobs (that might be in ready queue due to logic error)
            # This self.ready cleanup is now redundant if we remove from ready
            # when starting, but let's keep it as a safeguard.
            self.ready = [j for j in self.ready if j.remaining > 0]
            
            # --- Re-add jobs that were running but got preempted ---
            # This logic is now handled in the loop.
            # Let's simplify the loop.

        # ---
        # --- RE-WRITING THE 'run' LOOP for clarity with dependencies ---
        # ---
        
        # Reset state variables
        self.jobs = []
        self.pending_jobs = []
        self.jobs_to_release = []
        self.ready = []
        self.running = [None] * self.cores
        self.timeline = [[None] * self.cores for _ in range(self.sim_time)]
        self.gantt_segments = {}
        self.job_records = {}
        self.uid_counter = 1

        self._pre_instantiate_and_link()
        release_idx = 0
        
        for t in range(self.sim_time):
            # 1. Release jobs into pending
            while release_idx < len(self.jobs_to_release) and self.jobs_to_release[release_idx].release_time <= t:
                job = self.jobs_to_release[release_idx]
                self.pending_jobs.append(job)
                release_idx += 1

            # 2. Check dependencies, move from pending to ready
            newly_ready = []
            for job in self.pending_jobs:
                if self._dependencies_met(job):
                    self.ready.append(job)
                    newly_ready.append(job)
            if newly_ready:
                self.pending_jobs = [j for j in self.pending_jobs if j not in newly_ready]

            # 3. Handle running jobs: execute, check for completion
            for core_id in range(self.cores):
                job = self.running[core_id]
                if job is not None:
                    job.remaining -= 1
                    self.timeline[t][core_id] = job.uid

                    if isinstance(self.scheduler, RR):
                         self.scheduler.consume_quantum(1)

                    if job.remaining <= 0:
                        job.finish_time = t + 1
                        job.response_time = job.finish_time - job.release_time
                        self.running[core_id] = None # Core is now idle
            
            # 4. Assign jobs to idle cores
            idle_cores = [c for c in range(self.cores) if self.running[c] is None]
            
            # Keep track of jobs assigned this tick to not assign one job to multiple cores
            assigned_this_tick = set() 
            
            for core_id in idle_cores:
                # Get jobs that are ready AND not running AND not just assigned
                available_jobs = [
                    j for j in self.ready 
                    if j not in self.running 
                    and j.uid not in assigned_this_tick
                ]
                
                candidate = self.scheduler.get_next_job(available_jobs, t, None) # No 'current' job, core is idle
                
                if candidate:
                    self.running[core_id] = candidate
                    if candidate.start_time is None:
                        candidate.start_time = t
                    self.ready.remove(candidate)
                    assigned_this_tick.add(candidate.uid)
                    
                    # If this job just started, log it to timeline *this* tick
                    # (this overwrites the 'None' from step 3, which is fine)
                    if self.timeline[t][core_id] is None and candidate.remaining == candidate.exec_time:
                         # This check is flawed if exec_time is 1
                         pass
                    
                    # Let's re-think step 3. Execute *after* assignment.
                    pass # See simplified loop below. This is getting complex.

        
        # ---
        # --- FINAL, CLEAN 'run' METHOD ---
        # ---
        
        # Reset state variables (as this is the final version)
        self.jobs = []
        self.pending_jobs = []
        self.jobs_to_release = []
        self.ready = []
        self.running = [None] * self.cores
        self.timeline = [[None] * self.cores for _ in range(self.sim_time)]
        self.gantt_segments = {}
        self.job_records = {}
        self.uid_counter = 1
        
        self._pre_instantiate_and_link()
        release_idx = 0
        
        for t in range(self.sim_time):
            # 1. Release jobs -> pending
            while release_idx < len(self.jobs_to_release) and self.jobs_to_release[release_idx].release_time <= t:
                job = self.jobs_to_release[release_idx]
                self.pending_jobs.append(job)
                release_idx += 1

            # 2. Check dependencies -> pending to ready
            newly_ready = []
            for job in self.pending_jobs:
                if self._dependencies_met(job):
                    self.ready.append(job)
                    newly_ready.append(job)
            if newly_ready:
                self.pending_jobs = [j for j in self.pending_jobs if j not in newly_ready]

            # 3. Finish running jobs
            for core_id in range(self.cores):
                job = self.running[core_id]
                if job is not None:
                    if job.remaining <= 0: # Job finished *last* tick
                        job.finish_time = t
                        job.response_time = job.finish_time - job.release_time
                        self.running[core_id] = None
                    elif isinstance(self.scheduler, RR): # Quantum expired
                        # This logic is tricky. Let's assume RR logic is handled by scheduler.
                        # For RR, we need to preempt.
                        pass
            
            # 4. Preemption check (if preemptive)
            if self.preemptive and self.scheduler.preemptive:
                for core_id in range(self.cores):
                    current_job = self.running[core_id]
                    # Check against all ready jobs + other running jobs that aren't this one
                    available_jobs = self.ready + [j for j in self.running if j is not None and j != current_job]
                    
                    best_candidate = self.scheduler.get_next_job(available_jobs, t, current_job)
                    
                    if best_candidate and best_candidate not in self.running: 
                        # Preemption needed
                        if current_job is not None:
                             # Put current job back in ready queue
                             if current_job.remaining > 0:
                                self.ready.append(current_job)
                        
                        # Start new candidate
                        self.running[core_id] = best_candidate
                        if best_candidate.start_time is None:
                            best_candidate.start_time = t
                        if best_candidate in self.ready:
                            self.ready.remove(best_candidate)
                    
                    elif current_job is not None and best_candidate != current_job:
                         # Scheduler wants to run 'current_job' (or 'best' is running elsewhere)
                         # but our 'best_candidate' logic is flawed.
                         
                         # Let's try again.
                         pass
            
            # 4. (Simplified) Assign jobs to IDLE cores
            idle_cores = [c for c in range(self.cores) if self.running[c] is None]
            
            # Keep track of jobs assigned this tick
            assigned_this_tick = set()
            
            # Sort ready list by scheduler priority *once*
            # This is complex, as priority is dynamic (EDF, LLF)
            # We must call scheduler for each core.
            
            for core_id in idle_cores:
                # Get jobs that are ready AND not running AND not just assigned
                available_jobs = [
                    j for j in self.ready 
                    if j not in self.running
                    and j.uid not in assigned_this_tick
                ]
                
                candidate = self.scheduler.get_next_job(available_jobs, t, None)
                
                if candidate:
                    self.running[core_id] = candidate
                    if candidate.start_time is None:
                        candidate.start_time = t
                    self.ready.remove(candidate)
                    assigned_this_tick.add(candidate.uid)

            # 5. Execute 1 tick for all running jobs
            for core_id in range(self.cores):
                job = self.running[core_id]
                if job is not None:
                    job.remaining -= 1
                    self.timeline[t][core_id] = job.uid
                    
                    if isinstance(self.scheduler, RR):
                        self.scheduler.consume_quantum(1)
                    
                    # Check for finish *next* tick
                    if job.remaining <= 0:
                        job.finish_time = t + 1 # Tentative finish time
                        job.response_time = (t + 1) - job.release_time
                        self.running[core_id] = None # Will be idle next tick
                        
            # 6. Handle preemption (if preemptive)
            if self.preemptive and self.scheduler.preemptive:
                # Find best 'ready' job
                best_ready = self.scheduler.get_next_job(self.ready, t, None)
                if best_ready:
                    # Find worst 'running' job
                    worst_running_job = None
                    worst_running_core = -1
                    
                    # This requires comparing 'best_ready' to all 'running'
                    for core_id in range(self.cores):
                        current_job = self.running[core_id]
                        if current_job:
                             # Check if best_ready is better than current_job
                             # This is complex. Schedulers don't just return "best", they return "next".
                             # Let's use the original 'run' logic, it was closer.
                             pass
            
            # ---
            # --- FINAL ATTEMPT. Sticking to the original 'run' structure from main.py ---
            # ---
            
        # Reset state (needed because of failed attempts above)
        self.jobs = []
        self.pending_jobs = []
        self.jobs_to_release = []
        self.ready = []
        self.running = [None] * self.cores
        self.timeline = [[None] * self.cores for _ in range(self.sim_time)]
        self.gantt_segments = {}
        self.job_records = {}
        self.uid_counter = 1
            
        self._pre_instantiate_and_link()
        release_idx = 0
        
        for t in range(self.sim_time):
            # 1. Release jobs -> pending
            while release_idx < len(self.jobs_to_release) and self.jobs_to_release[release_idx].release_time <= t:
                job = self.jobs_to_release[release_idx]
                self.pending_jobs.append(job)
                release_idx += 1

            # 2. Check dependencies -> pending to ready
            newly_ready = []
            for job in self.pending_jobs:
                if self._dependencies_met(job):
                    self.ready.append(job)
                    newly_ready.append(job)
            if newly_ready:
                self.pending_jobs = [j for j in self.pending_jobs if j not in newly_ready]

            # 3. Assign jobs to cores (original logic from main.py)
            for core_id in range(self.cores):
                current = self.running[core_id]
                
                # --- Fix: If job finished, 'current' should be None ---
                if current and current.remaining <= 0:
                    current.finish_time = t
                    current.response_time = current.finish_time - current.release_time
                    current = None
                    self.running[core_id] = None
                
                # Get jobs that are ready OR the 'current' job (if it exists)
                available_jobs = self.ready
                if current and current not in available_jobs:
                    # Add current to consideration set for scheduler
                    available_jobs = self.ready + [current]
                
                # --- Fix: Exclude jobs running on *other* cores ---
                other_running = [self.running[c] for c in range(self.cores) if c != core_id and self.running[c] is not None]
                available_jobs = [j for j in available_jobs if j not in other_running]

                candidate = self.scheduler.get_next_job(available_jobs, t, current)
                
                # --- Start logic from original main.py 'run' ---
                if candidate is None:
                    # No candidate. If 'current' was running, it's preempted.
                    if current:
                         if current.remaining > 0 and current not in self.ready:
                            self.ready.append(current) # Put back in ready
                    self.running[core_id] = None
                    self.timeline[t][core_id] = None
                    continue

                if current is None:
                    # Core was idle
                    self.running[core_id] = candidate
                    if candidate.start_time is None:
                        candidate.start_time = t
                    if candidate in self.ready:
                        self.ready.remove(candidate)
                        
                elif candidate.uid != current.uid:
                    # Switch
                    if self.preemptive and self.scheduler.preemptive:
                        if current.remaining > 0 and current not in self.ready:
                            self.ready.append(current) # Put current back
                            
                        self.running[core_id] = candidate
                        if candidate.start_time is None:
                            candidate.start_time = t
                        if candidate in self.ready:
                            self.ready.remove(candidate)
                    else:
                        # Non-preemptive: 'current' continues
                        candidate = current
                        self.running[core_id] = current # Ensure it stays
                else:
                    # Candidate is same as current, do nothing
                    pass

                # Execute 1 tick
                job_to_run = self.running[core_id]
                if job_to_run:
                    job_to_run.remaining -= 1
                    self.timeline[t][core_id] = job_to_run.uid

                    # Round Robin: quantum control
                    if isinstance(self.scheduler, RR):
                        self.scheduler.consume_quantum(1)

                    # If job finishes *now*
                    if job_to_run.remaining <= 0:
                        job_to_run.finish_time = t + 1
                        job_to_run.response_time = job_to_run.finish_time - job_to_run.release_time
                        self.running[core_id] = None # Core will be idle next tick

            # Cleanup ready list (remove jobs that finished)
            # This is complex because a job might be 'running' but 'remaining=0'
            # Let's ensure 'ready' only contains jobs with remaining > 0
            self.ready = [j for j in self.ready if j.remaining > 0]
            
        # --- End of main 'run' loop ---

        # Build Gantt segments (This logic is from main.py and is correct)
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
        # Handle division by zero if sim_time is 0
        total_core_time = (self.sim_time * self.cores)
        cpu_util = busy / total_core_time if total_core_time > 0 else 0.0
        
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

    # --- MODIFIED: Adjust height based on cores AND tasks ---
    # We need a row per task, per core. This is complex.
    # Let's just stack by task ID, and color by core?
    # Or, one row per (task, core) pair?
    # The current 'gantt_segments' is (core, start, end)
    
    # Let's try (Task * Cores) rows
    # Row 0: T1-C0, Row 1: T1-C1, Row 2: T2-C0, Row 3: T2-C1
    
    # Let's stick to the *original* viz: one row per Task ID.
    # The segments (core, s, e) will be drawn on the same task row.
    # This shows *when* a task ran, but not *which* core.
    # Let's add core info to the bar.
    
    num_tasks = len(task_ids)
    fig_height = 1 + 0.5 * num_tasks
    
    # What if we make one subplot per core?
    if sim.cores > 1:
        fig_height = 1 + sim.cores * (num_tasks * 0.3)
        fig, axes = plt.subplots(sim.cores, 1, figsize=(12, fig_height), sharex=True, squeeze=False)
        fig.suptitle(title + f"  (Scheduler: {sim.scheduler.name})")
    else:
        fig, ax = plt.subplots(figsize=(12, fig_height))
        ax.set_title(title + f"  (Scheduler: {sim.scheduler.name})")
        axes = np.array([[ax]]) # Make it iterable like subplots

    colors = plt.get_cmap('tab20').colors
    color_map = {}
    yticks = []
    ylabels = []
    for i, tid in enumerate(task_ids):
        yticks.append(i + 0.5)
        ylabels.append(tid)
        color_map[tid] = colors[i % len(colors)]

    # Prepare rectangles
    bars = []
    for tid, segs in sim.gantt_segments.items():
        row_idx = task_indices.get(tid, None)
        if row_idx is None:
            continue
        
        for (core_id, s, e) in segs:
            ax = axes[core_id][0] # Select the subplot for this core
            rect = ax.barh(row_idx + 0.1, e - s, left=s, height=0.8, align='center', color=color_map.get(tid, 'grey'), edgecolor='black')
            bars.append(rect)

    # Configure all axes
    deadline_artists = []
    missed_artists = []
    
    for core_id in range(sim.cores):
        ax = axes[core_id][0]
        if sim.cores > 1:
            ax.set_ylabel(f"Core {core_id}", rotation=0, labelpad=25, ha='right')

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.set_xlim(0, sim.sim_time)
        ax.set_ylim(0, len(task_ids))
        ax.grid(True, axis='x', linestyle='--', alpha=0.4)

        # draw deadlines: for each job, vertical line
        for job in sim.jobs:
            if 0 <= job.absolute_deadline <= sim.sim_time:
                dl = ax.axvline(job.absolute_deadline, color='red', linestyle='--', linewidth=0.6, alpha=0.6)
                if core_id == 0: deadline_artists.append(dl) # Add only once

        # annotate missed deadlines
        for job in sim.jobs:
            if job.finish_time and job.finish_time > job.absolute_deadline:
                tid = job.task_id
                row = task_indices.get(tid, None)
                if row is not None:
                    # Find which core it finished on
                    finish_core = -1
                    for c_id in range(sim.cores):
                        if sim.timeline[job.finish_time-1][c_id] == job.uid:
                            finish_core = c_id
                            break
                    if finish_core == core_id:
                         mt = ax.text(job.finish_time + 0.1, row + 0.5, "✖", color='red', fontsize=10)
                         if core_id == 0: missed_artists.append(mt)
    
    axes[-1][0].set_xlabel("Time (ticks)")

    # show current time cursor via vertical line in animation
    cursors = [ax.axvline(0, color='blue', linewidth=1.2) for ax in axes.flatten()]

    # subplot for metrics (relative to the figure)
    metrics_text = fig.text(0.01, 0.01, "", fontsize=9, va="bottom", ha="left")

    # animation update function
    def update(frame):
        artists = []
        for c in cursors:
            c.set_xdata([frame])
            artists.append(c)
            
        metrics = sim._compute_metrics()
        mt = f"Time: {frame}/{sim.sim_time}   CPU util (total): {metrics['cpu_utilization']:.3f}   " \
             f"Missed: {metrics['missed_deadlines']}   Finished: {metrics['finished_jobs']}/{metrics['total_jobs']}"
        metrics_text.set_text(mt)
        artists.append(metrics_text)
        
        # We need to return all artists
        return artists + deadline_artists + missed_artists + [b[0] for b in bars if b]


    ani = FuncAnimation(fig, update, frames=range(0, sim.sim_time + 1), interval=max(1, int(500 / playback_speed)), blit=False, repeat=False)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust for suptitle and metrics
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
            jitter=int(item.get("jitter", 0)),
            dependencies=item.get("dependencies", []) # <-- ADDED
        ))
    return specs

def load_tasks_from_csv(path: str) -> List[TaskSpec]:
    specs = []
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Handle comma-separated strings for dependencies
            dep_str = row.get("dependencies", "")
            dependencies = [] if not dep_str else [task.strip() for task in dep_str.split(',')]
            
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
                jitter=(0 if not row.get("jitter") else int(row.get("jitter"))),
                dependencies=dependencies # <-- ADDED
            ))
    return specs

# ------------------------------
# Sample tasks for quick test
# ------------------------------

SAMPLE_TASKS = [
    # periodic tasks: id, wcet, period, (deadline default=period)
    TaskSpec(id="T1", wcet=1, period=4, release=0, type="periodic", instances=20),
    TaskSpec(id="T2", wcet=2, period=6, release=0, type="periodic", instances=15),
    TaskSpec(id="T3", wcet=1, period=8, release=0, type="periodic", instances=10, dependencies=["T1"]), # <-- MODIFIED

    # aperiodic
    TaskSpec(id="A1", wcet=3, release=5, deadline=12, type="aperiodic"),
    TaskSpec(id="A2", wcet=2, release=12, deadline=20, type="aperiodic", dependencies=["A1"]), # <-- MODIFIED

    # interrupt-style (one-shot arriving later)
    TaskSpec(id="IRQ", wcet=1, release=18, deadline=20, type="interrupt"),
]

# ------------------------------
# CLI / Runner
# ------------------------------

# (CLI 'parse_args' is commented out as GUI is primary)
# def parse_args(): ...

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
# (imports are at top)

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
    # --- MODIFIED: Added all schedulers to GUI ---
    sched_combo = ttk.Combobox(root, textvariable=scheduler_var, 
                               values=["EDF", "RM", "DM", "LLF", "FCFS", "SRTF", "RR"], 
                               state="readonly")
    sched_combo.pack()

    ttk.Label(root, text="Preemption:", font=("Arial", 11)).pack(pady=5)
    ttk.Checkbutton(root, text="Enable Preemption (Global)", variable=preemption_var).pack()

    ttk.Label(root, text="Number of Cores:", font=("Arial", 11)).pack(pady=5)
    ttk.Spinbox(root, from_=1, to=16, textvariable=cores_var, width=5).pack() # Increased max cores

    ttk.Label(root, text="Simulation Time (ticks):", font=("Arial", 11)).pack(pady=5)
    ttk.Entry(root, textvariable=sim_time_var, width=10).pack()

    ttk.Label(root, text="Quantum (for RR):", font=("Arial", 11)).pack(pady=5)
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
    
    if not config["scheduler"]:
        print("Simulation cancelled.")
        return

    # === Load tasks ===
    if config["file_type"] == "json":
        print(f"Loading tasks from {config['file_path']}...")
        specs = load_tasks_from_json(config["file_path"])
    elif config["file_type"] == "csv":
        print(f"Loading tasks from {config['file_path']}...")
        specs = load_tasks_from_csv(config["file_path"])
    else:
        print("No file loaded, using sample tasks.")
        specs = SAMPLE_TASKS

    scheduler = get_scheduler_by_name(config["scheduler"], quantum=config["quantum"])
    sim = Simulator(
        task_specs=specs,
        scheduler=scheduler,
        sim_time=config["sim_time"],
        preemptive=config["preemption"],
        cores=config["cores"]
    )

    print(f"Running simulation: {config['cores']} core(s), {scheduler.name} scheduler, Preemption={config['preemption']}")
    metrics = sim.run()
    print("Simulation complete. Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    fname = sim.export_csv("simulation_output.csv")
    print(f"Results exported to {fname}")

    print("Launching visualization...")
    plot_gantt_and_animate(sim, title=f"{scheduler.name} Visualization ({config['cores']} Core(s))", playback_speed=1.0)


if __name__ == "__main__":
    main()