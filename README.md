# Local LLM Tracing and Replay Platform

A lightweight C++ project that simulates how a local transformer/LLM can be traced, inspected, and visualized through a terminal dashboard.

This repository focuses on:
- layer-wise telemetry capture
- fixed-size ring-buffer logging
- attention visualization
- runtime metrics inspection
- anomaly tracking
- replay of stored trace sessions
- keyboard-driven terminal dashboard

## Project Overview

This project was built as a tracing and diagnostics platform for understanding how data flows through a transformer-style model.

The system is designed to:
- hook into a model execution pipeline in a non-invasive way
- capture layer-level metadata such as tensor shape, latency, sparsity, mean, and max activation
- visualize traces live in a terminal-based dashboard
- replay previously saved sessions
- highlight possible numerical anomalies

## Final Implementation Choice

During development, I explored two dashboard/tracing approaches:

1. **A modular `llmscope` architecture**  
   This version split the project into separate `core`, `engine`, and `tui` layers.

2. **The legacy dashboard-based implementation**  
   This version was cleaner, more stable, and easier to demonstrate live.

For the final submission, I chose the **legacy dashboard-based version** because it produced a smoother demo, fewer build issues, and a clearer visualization flow.

The modular `llmscope` approach was useful as an experiment, but it was not retained in the final demo branch to avoid confusion and keep the repository focused on one stable implementation.

## Features

### Core tracing
- fixed-size ring buffer for trace events
- tracer for layer/submodule events
- metadata capture for:
  - event id
  - token id
  - layer name
  - submodule name
  - tensor shape
  - dtype
  - timestamp
  - latency
  - sparsity
  - mean activation
  - max activation
  - anomaly flag

### Interactive dashboard
- model topology panel
- live packet stream panel
- attention matrix visualizer
- runtime metrics inspector
- anomaly ledger
- keyboard navigation
- focus switching with `Tab`
- layer selection with `j/k`
- matrix navigation with `h/j/k/l`
- contrast adjustment with `+/-`
- fullscreen toggle with `F`
- quit with `Q`

### Replay engine
- save trace sessions
- load old trace logs
- replay past sessions inside the dashboard

## Dashboard Overview

### 1. Model Topology
Shows the layer structure and lets you select the active capture target.

### 2. Live Packet Stream
Displays recent trace events in a compact stream.

### 3. Attention Matrix Visualizer
Shows a token-by-token attention-style view for the selected layer.

### 4. Runtime Metrics Inspector
Displays:
- tensor shape
- dtype
- sparsity rate
- latency
- mean activation
- max activation

### 5. Numerical Anomaly Ledger
Lists suspicious events such as:
- high max activation
- unusually high latency
- high sparsity
- NaN/Inf-like behavior

## Screenshots

### Traceformer dashboard
![Traceformer dashboard 1](Screenshots/dashboard1/traceformer1.png)
![Traceformer dashboard 2](Screenshots/dashboard1/traceformer2.png)

### Alternate dashboard (LLMScope)
![LLMScope dashboard 1](Screenshots/dashboard2/llmscope1.png)
![LLMScope dashboard 2](Screenshots/dashboard2/llmscope2.png)

## Project Structure

```text
traceformer/
├── include/
│   ├── attention.hpp
│   ├── dashboard.hpp
│   ├── metrics.hpp
│   ├── replay.hpp
│   ├── ring_buffer.hpp
│   ├── tracer.hpp
│   └── tui.hpp
├── src/
│   ├── attention.cpp
│   ├── dashboard.cpp
│   ├── main.cpp
│   ├── metrics.cpp
│   ├── replay.cpp
│   ├── ring_buffer.cpp
│   ├── tracer.cpp
│   └── tui.cpp
├── Screenshots/
│   ├── dashboard1/
│   └── dashboard2/
├── CMakeLists.txt
