// <copyright file="ISweepablePipelineNode.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Sweeper;

namespace MLNet.AutoPipeline
{
    public enum SweepablePipelineNodeType
    {
        /// <summary>
        /// Sweepable node Type.
        /// </summary>
        Sweepable = 0,

        /// <summary>
        /// Unsweepable node type.
        /// </summary>
        Unsweeapble = 1,
    }

    public interface ISweepablePipelineNode
    {
        IEstimator<ITransformer> BuildEstimator(ParameterSet parameters = null);

        TransformerScope Scope { get; }

        IValueGenerator[] ValueGenerators { get; }

        SweepablePipelineNodeType NodeType { get; }

        string Summary();

        string EstimatorName { get; }
    }
}
