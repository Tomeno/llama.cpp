# EdgeRock verification tools, see docs/edgerock.md
foreach(name dump logits state-test)
    set(TARGET llama-edgerock-${name})
    add_executable(${TARGET} edgerock-${name}.cpp)
    target_link_libraries(${TARGET} PRIVATE llama ggml ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(${TARGET} PRIVATE cxx_std_17)
endforeach()
