cmake_minimum_required(VERSION 3.14)
project(traceformer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(traceformer
    src/main.cpp
    src/tui.cpp
    src/ring_buffer.cpp
    src/metrics.cpp
    src/attention.cpp
    src/dashboard.cpp
    src/tracer.cpp
)