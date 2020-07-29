// <copyright file="IEstimatorNode.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Sweeper;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace MLNet.AutoPipeline
{
    public enum EstimatorNodeType
    {
        /// <summary>
        /// Single Node.
        /// A single node in EstimatorNodeChain
        /// Can be IEstimator/EstimatorChain/AutoEstimator.
        /// </summary>
        Node = 0,

        /// <summary>
        /// AutoEstimatorChain. List of Node|NodeGroup.
        /// </summary>
        NodeChain = 1,

        /// <summary>
        /// Group of Node|NodeChain.
        /// </summary>
        NodeGroup = 2,
    }

    public interface IEstimatorNode
    {
        IEnumerable<SweepablePipeline> BuildSweepablePipelines();

        EstimatorNodeType NodeType { get; }

        string Summary();
    }
}
