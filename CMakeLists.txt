cmake_minimum_required(VERSION 2.8.3)
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

################################################
## Declare ROS messages, services and actions ##
################################################

## Messages defined in deformable_manipulation_msgs package

###################################
## catkin specific configuration ##
###################################

catkin_package(
    INCLUDE_DIRS    include
    LIBRARIES       ${PROJECT_NAME}
    CATKIN_DEPENDS  ${CATKIN_PACKAGES}
    DEPENDS         Eigen3 FLANN
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
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -O3")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Og")# -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Wextra -Wpedantic -Wconversion -Werror=return-type -Werror=missing-field-initializers -DEIGEN_DONT_PARALLELIZE")

unset(USE_OPEN_MP CACHE)
option(USE_OPEN_MP "Set to ON if we want to use OpenMP" ON)
if(USE_OPEN_MP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DUSE_OPEN_MP")
    message(STATUS "Enabling OpenMP for SMMAP library")
else(USE_OPEN_MP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    message(WARNING "Disabling OpenMP for SMMAP library")
endif(USE_OPEN_MP)


## Library file for the whole project`
FILE(GLOB hpp_SRC "include/${PROJECT_NAME}/*.hpp")
FILE(GLOB h_SRC "include/${PROJECT_NAME}/*.h")
FILE(GLOB cpp_SRC "src/*.cpp")

add_library(${PROJECT_NAME} ${hpp_SRC} ${h_SRC} ${cpp_SRC})
add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${FLANN_LIBRARIES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} ${FLANN_DEFINITIONS}")

add_executable(planner_test_node nodes/planner_test_node.cpp)
add_dependencies(planner_test_node ${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})
target_link_libraries(planner_test_node ${PROJECT_NAME} ${catkin_LIBRARIES})

add_executable(transition_learning_data_generation_node nodes/transition_learning_data_generation_node.cpp)
add_dependencies(transition_learning_data_generation_node ${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})
target_link_libraries(transition_learning_data_generation_node ${PROJECT_NAME} ${catkin_LIBRARIES})

## Synthethic trials
#add_executable(kalman_filter_synthetic_trials nodes/kalman_filter_synthetic_trials.cpp src/cvxopt_solvers.cpp src/gurobi_solvers.cpp)
#add_dependencies(kalman_filter_synthetic_trials ${catkin_EXPORTED_TARGETS})
#target_link_libraries(kalman_filter_synthetic_trials ${catkin_LIBRARIES} ${PYTHON_LIBRARIES} ${GUROBI_LIBRARIES})
#set_target_properties(kalman_filter_synthetic_trials PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS}")

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#     scripts/my_python_script
#     DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS smmap smmap_node
#     ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#     LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#     RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#     DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#     FILES_MATCHING PATTERN "*.h"
#     PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#     # myfile1
#     # myfile2
#     DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_smmap.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#     target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
