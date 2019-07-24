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
    tf2_ros
    tf2_eigen
)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS ${CATKIN_PACKAGES})

## System dependencies are found with CMake's conventions
find_package(cmake_modules REQUIRED)
find_package(Eigen3 REQUIRED)
set(Eigen3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})



set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")
find_package(Flann REQUIRED)
find_package(SVM REQUIRED)

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
    DEPENDS         Eigen3 FLANN SVM
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
    ${FLANN_INCLUDE_DIRS}
    ${SVM_INCLUDE_DIRS}
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

############# Infrastructure ###################################################

add_custom_target(${PROJECT_NAME}_infrastructure_headers SOURCES
    include/${PROJECT_NAME}/point_reflector.hpp
    include/${PROJECT_NAME}/robot_interface.h
    include/${PROJECT_NAME}/ros_communication_helpers.h
    include/${PROJECT_NAME}/task_function_pointer_types.h
    include/${PROJECT_NAME}/task_specification.h
    include/${PROJECT_NAME}/task_specification_implementions.h
    include/${PROJECT_NAME}/trajectory.hpp
)

add_library(${PROJECT_NAME}_infrastructure
    src/robot_interface.cpp
    src/ros_communication_helpers.cpp
    src/task_specification.cpp
    src/task_specification_implementations.cpp
    src/trajectory.cpp
)
add_dependencies(${PROJECT_NAME}_infrastructure ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_infrastructure ${catkin_LIBRARIES})

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

add_library(${PROJECT_NAME}_controllers
    include/${PROJECT_NAME}/deformable_controller.h
    include/${PROJECT_NAME}/least_squares_stretching_constraint_controller.h
    include/${PROJECT_NAME}/straight_line_controller.h
    include/${PROJECT_NAME}/stretching_constraint_controller.h
    include/${PROJECT_NAME}/least_squares_controller_with_object_avoidance.h
    src/least_squares_controller_with_object_avoidance.cpp
    src/stretching_constraint_controller.cpp
    src/straight_line_controller.cpp
    src/least_squares_stretching_constraint_controller.cpp
)
add_dependencies(${PROJECT_NAME}_controllers ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_controllers ${catkin_LIBRARIES})

############# Transitions ######################################################

add_custom_target(${PROJECT_NAME}_transitions_headers SOURCES
    include/${PROJECT_NAME}/transition_estimation.h
    include/${PROJECT_NAME}/quinlan_rubber_band.h
)

add_library(${PROJECT_NAME}_transitions
    src/transition_estimation.cpp
    src/quinlan_rubber_band.cpp
)
add_dependencies(${PROJECT_NAME}_transitions ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_transitions ${catkin_LIBRARIES} ${SVM_LIBRARIES})

############# Planning #########################################################

add_custom_target(${PROJECT_NAME}_planning_headers SOURCES
    include/${PROJECT_NAME}/band_rrt.h
    include/${PROJECT_NAME}/task_framework.h
)

add_library(${PROJECT_NAME}_planning
    src/band_rrt.cpp
    src/task_framework.cpp
)
add_dependencies(${PROJECT_NAME}_planning ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_planning
    ${PROJECT_NAME}_infrastructure
    ${PROJECT_NAME}_models
    ${PROJECT_NAME}_controllers
    ${PROJECT_NAME}_transitions
    ${catkin_LIBRARIES}
    ${FLANN_LIBRARIES}
)
set_target_properties(${PROJECT_NAME}_planning PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${FLANN_DEFINITIONS}")

add_executable(planner_test_node nodes/planner_test_node.cpp)
add_dependencies(planner_test_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(planner_test_node ${PROJECT_NAME}_planning ${catkin_LIBRARIES})

############# Learning Testing #################################################

add_custom_target(${PROJECT_NAME}_learning_testing_headers SOURCES
    include/${PROJECT_NAME}/transition_learning_data_generation.h
)

add_library(${PROJECT_NAME}_learning_testing
    src/transition_learning_data_generation.cpp
)
add_dependencies(${PROJECT_NAME}_learning_testing ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME}_learning_testing
#    ${PROJECT_NAME}_infrastructure
#    ${PROJECT_NAME}_transitions
    ${PROJECT_NAME}_planning
    ${catkin_LIBRARIES}
)

add_executable(transition_learning_data_generation_node nodes/transition_learning_data_generation_node.cpp)
add_dependencies(transition_learning_data_generation_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(transition_learning_data_generation_node ${PROJECT_NAME}_learning_testing ${catkin_LIBRARIES})
