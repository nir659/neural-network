cmake_minimum_required(VERSION 3.20)
project(neural-network LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB LIB_SOURCES "src/*.cpp")

add_executable(neural_net main.cpp ${LIB_SOURCES})
target_include_directories(neural_net PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_tensor
  tests/test_tensor.cpp
  ${LIB_SOURCES}
)
target_include_directories(test_tensor PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_network
  tests/test_network.cpp
  ${LIB_SOURCES}
)
target_include_directories(test_network PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_gradient_check
  tests/test_gradient_check.cpp
  ${LIB_SOURCES}
)
target_include_directories(test_gradient_check PRIVATE ${CMAKE_SOURCE_DIR}/include)

enable_testing()
add_test(NAME tensor_tests  COMMAND test_tensor)
add_test(NAME network_tests COMMAND test_network)
add_test(NAME gradient_check_tests COMMAND test_gradient_check)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(neural_net    PRIVATE -O3 -march=native)
  target_compile_options(test_tensor   PRIVATE -O3 -march=native)
  target_compile_options(test_network  PRIVATE -O3 -march=native)
  target_compile_options(test_gradient_check PRIVATE -O3 -march=native)
endif()
