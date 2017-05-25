#ifndef RRT_HELPER_H
#define RRT_HELPER_H

#include <iostream>
#include <stdio.h>
#include <vector>


#include <arc_utilities/arc_helpers.hpp>
#include "smmap/virtual_rubber_band.h"


#define ERRORTHRESHOLD 0.002
#define GOALERROR 0.3
#define PI 3.141592653
#define STEPSIZE 0.3
#define SAMPLEBIAS 0.1
#define NUMITERATION 200
#define DOFCONFIG 6
#define STRETCHING_THRESHOLD 0

using namespace std;
using namespace Eigen;

namespace smmap {

    typedef std::vector<double> configuration;
    typedef std::vector<configuration> configSet;
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
        std::vector<double> lowerBound;
        std::vector<double> upperBound;
        std::vector<double> weight;

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


}






#endif
