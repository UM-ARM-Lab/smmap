#ifndef RRT_HELPER_H
#define RRT_HELPER_H

//#include <iostream>
//#include <stdio.h>
//#include <vector>


#include <arc_utilities/arc_helpers.hpp>
#include "smmap/virtual_rubber_band.h"

#include <arc_utilities/simple_rrt_planner.hpp>

#define ERRORTHRESHOLD 0.002
#define GOALERROR 0.3
#define PI 3.141592653
#define STEPSIZE 0.3
#define SAMPLEBIAS 0.1
#define NUMITERATION 200
#define DOFCONFIG 6
#define STRETCHING_THRESHOLD 0

namespace smmap {

    typedef std::pair<std::pair<Eigen::Affine3d, Eigen::Affine3d>, VirtualRubberBand> rrtConfig;

    namespace Rrt_function{

        // * nearest_neighbor_fn - given all nodes explored so far, and a new state, return the index of the "closest" node
        template<typename T, typename Allocator = std::allocator<T>>
        inline int64_t nearestNeighbor(
                std::vector<simple_rrt_planner::SimpleRRTPlannerState<T, Allocator>>& nodes,
                const T& config)
        {




            int64_t res;
            return res;
        }

        // const std::function<void(SimpleRRTPlannerState<T, Allocator>&, SimpleRRTPlannerState<T, Allocator>&)>& state_added_fn
        // * state_added_fn - callback function that takes (parent, child) for each extension
        template<typename T, typename Allocator = std::allocator<T>>
        inline void stateAdd(
                simple_rrt_planner::SimpleHybridRRTPlanner<T,Allocator>& parent,
                simple_rrt_planner::SimpleHybridRRTPlanner<T,Allocator>& child)
        {



        }


        // const std::function<bool(const T&)>& goal_reached_fn,
        // * goal_reached_fn - return if a given state meets the goal conditions (for example, within a radius of a goal state)
        // const std::function<void(SimpleRRTPlannerState<T, Allocator>&)>& goal_reached_callback_fn,
        // CAN USE DISTANCE FUNCTION INSTEAD


        // const std::function<T(void)>& sampling_fn,
        // * state_sampling_fn - returns a new state (randomly- or deterministically-sampled)
        template<typename T>
        inline T rrtSampling()
        {
            T randSample;


            return randSample;
        }


        // const std::function<std::vector<std::pair<T, int64_t>>(const T&, const T&)>& forward_propagation_fn,
        // * forward_propagation_fn - given the nearest neighbor and a new target state, returns the states that would grow the tree towards the target
        // * SHOULD : collosion checking, constraint violation checking
        // Determine the parent index of the new state
        // This process deserves some explanation
        // The "current relative parent index" is the index of the parent, relative to the list of propagated nodes.
        // A negative value means the nearest neighbor in the tree, zero means the first propagated node, and so on.
        // NOTE - the relative parent index *must* be lower than the index in the list of prograted nodes
        // i.e. the first node must have a negative value, and so on.
        template <typename T>
        inline std::vector<std::pair<T, int64_t>> takeStep(const T&, const T&)
        {
            std::vector<std::pair<T, int64_t>> propagation;



            return propagation;
        }


        // const std::function<bool(void)>& termination_check_fn
        // * termination_check_fn - returns if the planner should terminate (for example, if it has exceeded time/space limits)




    }












/*
    typedef Vector3d configuration;
    typedef VectorVector3d configSet;
    typedef std::pair<Eigen::Affine3d, Eigen::Affine3d> tConfigSet;

    typedef configSet* configSetPtr;
    typedef tConfigSet* tConfigSetPtr;

    configuration SE3toVconfig(Eigen::Affine3d tConfig);
    configSet SetSE3toVconfig(tConfigSet tConfigSetIn);

    Eigen::Affine3d VtoSE3config(configuration config);
    tConfigSet SetVtoSE3config(configSet config);

    class RRTNode;
    typedef RRTNode* RRTNodePtr;
    typedef std::vector<RRTNodePtr> tree;
    typedef tree* treePtr;

    //////////////////////////////////////////////////////////////////////////
    // RRTNode Class
    //////////////////////////////////////////////////////////////////////////

    class RRTNode
    {

    public:
        RRTNode();
        RRTNode(configSet configSetIn);
        ~RRTNode();

        void SetConfig(configSet config);
        configSet GetConfig();
        void AddStep(configSet s);

        void SetParent(RRTNodePtr parentNode);
        RRTNodePtr GetParent();

        // TODO: If it is not used later, delete this function (Euclidian Distance)
        float Distance(configuration B);
        bool SameNode(RRTNodePtr checkNode);

    private:
        configSet configSet_;
        RRTNodePtr parent_;
//        std::shared_ptr<VirtualRubberBand> virtual_rubber_band_between_grippers_;
    };

    //////////////////////////////////////////////////////////////////////////
    // Rrt Growing NodeTree Class
    //////////////////////////////////////////////////////////////////////////

    class NodeTree
    {
    public:
        NodeTree();
        NodeTree(RRTNodePtr initNode);
        ~NodeTree();

        RRTNodePtr GetRoot();

        void ResetTree();

        // In the implementation of the following functions, IsSame Node only depends
        // on the xyz position, I didn't take band's homotopic group into account.
        bool InTree(RRTNodePtr testNode);
        bool Add(RRTNodePtr growNode);
        bool Remove(RRTNodePtr remNode);
        RRTNodePtr GetNode(int index);
        int TreeSize();
        int GetIndex(RRTNodePtr findNode);
        bool IsRoot(RRTNodePtr checkNode);

        std::vector<RRTNodePtr> GetPath(RRTNodePtr goal);

    private:
        tree tree_;
        RRTNodePtr rootNode_;
    };

    typedef NodeTree* NodeTreePtr;

    ///////////////////////////////////////////////////////////////////////////
    // Parameters Set Class
    ///////////////////////////////////////////////////////////////////////////

    class ParameterSet
    {
    public:
        ParameterSet();

        // Geodesic of (i,j) gripper stored in i*n+j; n is num of grippers
//        void InitGeodesic(configSet initConfig, std::vector<RobotBasePtr> robots);

        void SetStart(tConfigSet startSE3);
        void SetStart(configSet start);

        void SetGoal(tConfigSet goalSE3);
        void SetGoal(configSet goal);

        // To simplify the problem, I assume all grippers having the same boundary
        Vector3d lowerBound;
        Vector3d upperBound;
        Vector3d weight;

        int numGrippers = 2;
//        configSet geodesicConfig;
//        std::vector<float> geodesic;

        float sampleBias = SAMPLEBIAS;
        float stepSize = STEPSIZE;
        int iteration = NUMITERATION;
        int isSmooth = 0;
        int isBiRRT = 0;

        configSet start_;
        configSet goal_;
        tConfigSet startSE3_;
        tConfigSet goalSE3_;
    };

    typedef NodeTree* NodeTreePtr;

    /////////////////////////////////////////////////////////////////////////////
    // Rrt Planner Class
    /////////////////////////////////////////////////////////////////////////////

    class RrtPlanner
    {
    public:

        RrtPlanner(std::pair<std::pair<Eigen::Affine3d, Eigen::Affine3d>, VirtualRubberBand> start_node,
                   std::pair<std::pair<Eigen::Affine3d, Eigen::Affine3d>, VirtualRubberBand> goal_node);
        RrtPlanner(std::pair<std::pair<Eigen::Affine3d, Eigen::Affine3d>, VirtualRubberBand> start_node,
                   std::pair<std::pair<Eigen::Affine3d, Eigen::Affine3d>, VirtualRubberBand> goal_node,
                   ParameterSet parameterIn);

        ///////////////////////////////////////////////////////////////////////////
        // Parameters input
        ///////////////////////////////////////////////////////////////////////////

        void SetParameters(ParameterSet parameterIn);
        ParameterSet GetParameters();

        ////////////////////////////////////////////////////////////////////////////
        // RRT planner helper function, for single configuration, usually not called from outside
        ////////////////////////////////////////////////////////////////////////////

        void ResetPath();

        int RandNode(int size);

        configuration RandSample();

        configuration ScaleConfig(configuration v, float scale);

        void ConfigPrintHelp(configuration config);

        bool OutBound(configuration a);

        bool SameDirection(configuration a, configuration b);

        configuration SumConfig(configuration A, configuration B);

        // A-B
        configuration SubtractConfig(configuration A, configuration B);

        // return unitstep vector, the step size is in #define, could be adjust later
        configuration UnitStep(configuration start, configuration goal, float stepSize);

        float Distance(configuration A, configuration B);  // Euclidian Distance

        float WeightedDis(configuration A, configuration B);


        ////////////////////////////////////////////////////////////////////////////
        // Helper function for configuration SET (Pair)
        ////////////////////////////////////////////////////////////////////////////

        void SetConfigPrintHelp(configSet config);

        bool SetOutBound(configSet a);

        configSet SetScaleConfig(configSet v, float scale);

//        configSet SetVecToConfig(std::vector<std::vector<double> > setVec);

//        std::vector<std::vector<double> > SetConfigToVec(configSet setConfig);

        float SetWeightedDis(configSet A, configSet B);

        configSet SetSumConfig(configSet A, configSet B);

        configSet SetSubtractConfig(configSet A, configSet B);  // A-B

        // Nearest Node on the tree
        RRTNode* NearestNode(configSet config, NodeTreePtr treePtr);

        configSet SampleSet(int gripperSize);

        configSet SampleSetWithBase(configuration baseSample, int index);

        configSet SetUnitStep(configSet start, configSet goal, float stepSize);

        // NOTE: Implementation of this function should be changed to use rubber band
        bool ConstraintViolation(configSet sampleConfig);

        // Set collision check helper
        bool SetCollisionCheck();

        ////////////////////////////////////////////////////////////////////////////
        // Main RRT Planning Function
        ////////////////////////////////////////////////////////////////////////////




    };
*/

}






#endif
