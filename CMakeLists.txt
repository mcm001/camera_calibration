cmake_minimum_required(VERSION 3.20)

set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "" FORCE)

project(camera_calibration)

add_subdirectory(Sleipnir)

find_package( OpenCV REQUIRED )

add_executable(camera_calibration src/main.cpp )
target_link_libraries(camera_calibration Sleipnir fmt::fmt ${OpenCV_LIBS}
)

# ew -- but we already have units in Sleipnir. Is there a less bad way to do this?
target_include_directories(camera_calibration PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Sleipnir/thirdparty/units/include
    ${OpenCV_INCLUDE_DIRS}
)
