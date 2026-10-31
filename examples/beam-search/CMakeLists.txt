set(TARGET llama-beam-search)
add_executable(${TARGET} beam-search.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama-common llama ggml ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ../../src) # llama-ext.h
target_compile_features(${TARGET} PRIVATE cxx_std_17)
