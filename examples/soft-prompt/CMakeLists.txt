# shared library with a C API, used by the Python scripts in this directory (ctypes)
set(TARGET llama-soft-prompt)
add_library(${TARGET} SHARED soft-prompt.cpp)
set_target_properties(${TARGET} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
                                           RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
install(TARGETS ${TARGET} LIBRARY)
target_link_libraries(${TARGET} PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
