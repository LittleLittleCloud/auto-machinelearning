// <copyright file="IAutoEstimatorNode.cs" company="BigMiao">
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
    public enum AutoEstimatorNodeType
    {
        /// <summary>
        /// Single Node.
        /// A single AutoEstimatorChainNode can be IEstimator/EstimatorChain/AutoEstimator.
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
    };

    public interface IAutoEstimatorNode
    {
        IEnumerable<ISingleNodeChain> BuildEstimatorChains();

        AutoEstimatorNodeType NodeType { get; }

        string Summary();
    }

    public interface ISingleNodeChain
    {
        IList<IValueGenerator> ValueGenerators { get; }

        IList<ISingleNodeBuilder> SingleNodeBuilders { get; }

        ISweeper Sweeper { get; }

        ISingleNodeChain Append(ISingleNodeBuilder builder);

        ISingleNodeChain Append<TTransformer>(IEstimator<TTransformer> estimator, TransformerScope scope = TransformerScope.Everything)
            where TTransformer : ITransformer;

        ISingleNodeChain Concat(ISingleNodeChain chain);

        string Summary();

        void UseSweeper(ISweeper sweeper);

        IEnumerable<EstimatorChain<ITransformer>> Sweeping(int maximum);
    }
}
