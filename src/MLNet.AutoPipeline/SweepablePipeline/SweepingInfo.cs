// <copyright file="SweepingInfo.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Sweeper;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    /// <summary>
    /// Sweeping info.
    /// </summary>
    public class SweepingInfo
    {
        public EstimatorChain<ITransformer> Pipeline { get; private set; }

        public ParameterSet Parameters { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="SweepingInfo"/> class.
        /// </summary>
        /// <param name="pipeline">Pipeline.</param>
        /// <param name="parameters">Parameters from <see cref="SweepablePipeline.Sweeper"/>.</param>
        public SweepingInfo(EstimatorChain<ITransformer> pipeline, ParameterSet parameters)
        {
            this.Pipeline = pipeline;
            this.Parameters = parameters;
        }
    }
}
