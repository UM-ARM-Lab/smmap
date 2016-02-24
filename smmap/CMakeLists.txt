cmake_minimum_required( VERSION 2.8.3 )
project( smmap )

set( CATKIN_PACKAGES
    actionlib
    arc_utilities
    geometry_msgs
    kinematics_toolbox
    image_transport
    cv_bridge
    ${PROJECT_NAME}_msgs
)

## Find catkin macros and libraries
find_package( catkin REQUIRED COMPONENTS ${CATKIN_PACKAGES} )

## System dependencies are found with CMake's conventions
find_package( cmake_modules REQUIRED )
find_package( Boost REQUIRED COMPONENTS filesystem )
find_package( Eigen REQUIRED )

################################################
## Declare ROS messages, services and actions ##
################################################

## Messages defined in smmap_msgs package

###################################
## catkin specific configuration ##
###################################

catkin_package(
    INCLUDE_DIRS    include
    LIBRARIES       ${PROJECT_NAME}
    CATKIN_DEPENDS  ${CATKIN_PACKAGES}
    DEPENDS         Eigen
                    Boost
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories( include
    SYSTEM
    ${Boost_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
)

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -g -O3 -march=native -Wall -Wextra -Wpedantic -Wconversion" )

unset( USE_OPEN_MP CACHE )
option( USE_OPEN_MP "Set to ON if we want to use OpenMP" ON )
if ( USE_OPEN_MP )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DUSE_OPEN_MP" )
    message( STATUS "Enabling OpenMP for SMMAP library" )
else( USE_OPEN_MP )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas" )
    message( WARNING "Disabling OpenMP for SMMAP library" )
endif( USE_OPEN_MP )


## Library file for the whole project
add_library( ${PROJECT_NAME}
    include/${PROJECT_NAME}/collision_helpers.hpp
    include/${PROJECT_NAME}/gripper_helpers.hpp
    include/${PROJECT_NAME}/ros_params.hpp
    include/${PROJECT_NAME}/ros_communication_helpers.hpp
    include/${PROJECT_NAME}/task.h
    src/task.cpp
    include/${PROJECT_NAME}/task_enums.h
    include/${PROJECT_NAME}/task_function_pointer_types.h
    include/${PROJECT_NAME}/task_specification.h
    include/${PROJECT_NAME}/task_specification_implementions.hpp
    src/task_specification.cpp
    include/${PROJECT_NAME}/point_reflector.hpp
    include/${PROJECT_NAME}/trajectory.hpp
    include/${PROJECT_NAME}/visualization_tools.h
    src/visualization_tools.cpp
    include/${PROJECT_NAME}/deformable_model.h
    src/deformable_model.cpp
    include/${PROJECT_NAME}/diminishing_rigidity_model.h
    src/diminishing_rigidity_model.cpp
    include/${PROJECT_NAME}/model_set.h
    src/model_set.cpp
    include/${PROJECT_NAME}/planner.h
    src/planner.cpp
)
add_dependencies( ${PROJECT_NAME}
    ${catkin_EXPORTED_TARGETS}
    ${PROJECT_NAME}_gencpp
)

## Test files to test specific functionality
add_executable( ${PROJECT_NAME}_test_node
    nodes/${PROJECT_NAME}_test_node.cpp
)
add_dependencies( ${PROJECT_NAME}_test_node
    ${PROJECT_NAME}
    ${catkin_EXPORTED_TARGETS}
)
target_link_libraries( ${PROJECT_NAME}_test_node
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
)

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
