cmake_minimum_required(VERSION 3.16)

project(traceformer_suite LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# ------------------------------------------------------------
# Legacy project (your current files stay untouched)
# ------------------------------------------------------------
add_executable(traceformer_legacy
    src/main.cpp
    src/tui.cpp
    src/ring_buffer.cpp
    src/metrics.cpp
    src/tracer.cpp
    src/attention.cpp
    src/dashboard.cpp
    src/replay.cpp
)

target_include_directories(traceformer_legacy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ------------------------------------------------------------
# New llmscope project
# ------------------------------------------------------------
add_library(llmscope_core
    src/core/tensor.cpp
    src/core/instrumentation.cpp
    src/core/model_graph.cpp
)

target_include_directories(llmscope_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(llmscope_engine
    src/engine/tokenizer.cpp
    src/engine/transformer.cpp
)

target_link_libraries(llmscope_engine PUBLIC llmscope_core)

add_executable(llm_scope
    src/modern_main.cpp
    src/tui/app.cpp
)

target_link_libraries(llm_scope PRIVATE llmscope_core llmscope_engine)
target_include_directories(llm_scope PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)