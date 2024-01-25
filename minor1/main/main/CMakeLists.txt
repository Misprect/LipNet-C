cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)

project(main)

set(CMAKE_PREFIX_PATH "W:\\Lib\\opencv\\build\\x64\\vc16\\lib")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_PREFIX_PATH "W:\\Lib\\libtorch")
find_package(Torch REQUIRED)

add_executable(CMakeTarget "main.h" "data.bin" "data.cpp")

target_link_libraries(CMakeTarget PRIVATE ${OpenCV_LIBS} "${TORCH_LIBRARIES}")

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET CMakeTarget
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:CMakeTarget>)
endif (MSVC)
