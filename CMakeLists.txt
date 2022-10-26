cmake_minimum_required(VERSION 3.1)
project(smmap)

set(CATKIN_PACKAGES
    actionlib
    arc_utilities
    smmap_utilities
    geometry_msgs
    kinematics_toolbox
    sdf_tools
    deformable_manipulation_msgs
    deformable_manipulation_experiment_params
#    pcl_ros
    tf2_ros
    tf2_eigen
)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS ${CATKIN_PACKAGES})

## System dependencies are found with CMake's conventions
find_package(cmake_modules REQUIRED)
find_package(Eigen3 REQUIRED)
set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

################################################
## Declare ROS messages, services and actions ##
################################################

## Messages defined in deformable_manipulation_msgs package

###################################
## catkin specific configuration ##
###################################

catkin_package(
    INCLUDE_DIRS    include
    LIBRARIES       ${PROJECT_NAME}_planning
    CATKIN_DEPENDS  ${CATKIN_PACKAGES}
    DEPENDS         EIGEN3
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include
    SYSTEM
    ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Og")# -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror=return-type -Werror=missing-field-initializers -DEIGEN_DONT_PARALLELIZE")

unset(USE_OPEN_MP CACHE)
option(USE_OPEN_MP "Set to ON if we want to use OpenMP" ON)
if(USE_OPEN_MP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DUSE_OPEN_MP")
    message(STATUS "Enabling OpenMP for SMMAP library")
else(USE_OPEN_MP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    message(WARNING "Disabling OpenMP for SMMAP library")
endif(USE_OPEN_MP)

############# Models ###########################################################

add_custom_target(${PROJECT_NAME}_models_headers SOURCES
    include/${PROJECT_NAME}/deformable_model.h
    include/${PROJECT_NAME}/jacobian_model.h
    include/${PROJECT_NAME}/adaptive_jacobian_model.h
    include/${PROJECT_NAME}/constraint_jacobian_model.h
    include/${PROJECT_NAME}/diminishing_rigidity_model.h
    include/${PROJECT_NAME}/least_squares_jacobian_model.h
)

add_library(${PROJECT_NAME}_models
    src/deformable_model.cpp
    src/jacobian_model.cpp
    src/adaptive_jacobian_model.cpp
    src/constraint_jacobian_model.cpp
    src/diminishing_rigidity_model.cpp
    src/least_squares_jacobian_model.cpp
)
add_dependencies(${PROJECT_NAME}_models ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_models ${catkin_LIBRARIES})