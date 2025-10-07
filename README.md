Project Proposal: Real-Time Task Timing Visualizer (Simulator)
By- Dhruv Belawat and Saarthak Gupta
1. Goal
Build a simulator that models real-time task scheduling on a CPU. The simulator will run task traces
(periodic, aperiodic, interrupts) and dynamically visualize scheduling decisions, deadlines, and
missed deadlines. It will support multiple real-time scheduling algorithms, allow dynamic task
arrivals, and provide interactive visualization.
2. Features
• Dynamic Task Arrivals
o Tasks can arrive at runtime (random arrivals, scripted from JSON/CSV, or user-added
via GUI).
• Scheduling Algorithms
o Real-time: Rate Monotonic (RM), Earliest Deadline First (EDF), Deadline Monotonic
(DM), Least Laxity First (LLF).
o OS-style: FCFS, Shortest Job First (SJF/SRTF), Round Robin (RR), Fixed Priority.
• Visualization
o Gantt-chart style timeline of task execution.
o Deadline markers, with missed deadlines highlighted.
o CPU idle periods shown distinctly.
• Metrics
o CPU utilization, response times, missed deadlines count.
o Export results to CSV.
3. Extensions
1. Energy-aware scheduling
o Simulate CPU energy states (active vs idle).
o Show estimated energy consumption under different schedulers.
2. Multiprocessor Scheduling
o Simulate dual-core or multi-core scheduling (Global EDF, Partitioned RM).
3. Task Dependencies
o Support Directed Acyclic Graphs (DAGs) to model precedence constraints.
4. Priority Inversion Simulation
o Model resource locking and priority inversion.
o Compare with Priority Inheritance protocol.
5. Jitter Modeling
o Introduce variability in task release times and visualize its impact.
4. Tools & Implementation Plan
• Language: Python (preferred for fast prototyping and visualization).
• Visualization:
o Matplotlib (Python desktop app).
o Optional: D3.js (web-based interactive version).
• Trace Input:
o Generated synthetic tasks.
o User interactive input (add task at runtime).
o Optional: load from CSV/JSON traces.
• System Requirements:
o Runs on a general-purpose laptop (Windows/Linux/Mac).
o No special hardware required.
5. Project Phases
1. Task Model & Simulator Core
o Implement task objects (arrival, exec, deadline, priority).
o Discrete time loop to simulate execution.
2. Scheduler Implementations
o RM, EDF, DM, LLF → baseline.
o Add OS schedulers for comparison (FCFS, RR, SJF).
3. Visualization Layer
o Gantt chart with deadlines, idle times, and missed deadlines.
4. Metrics & Reporting
o CPU utilization, response times, deadline misses → CSV export.
5. Extensions
o Energy-aware, multiprocessor, priority inversion, jitter, DAG dependencies.
6. Deliverables
• Simulator Software (Python app).
• Visualization Tool (timeline plots, interactive GUI).
• Documentation (design decisions, how to run, sample traces).
• Demo showing multiple scheduling algorithms on the same workload.
