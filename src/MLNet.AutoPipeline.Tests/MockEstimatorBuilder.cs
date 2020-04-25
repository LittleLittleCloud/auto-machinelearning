// <copyright file="MockEstimatorBuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using MLNet.AutoPipeline;
using MLNet.Sweeper;
using System.Collections.Generic;
using Xunit;
using Xunit.Abstractions;
using static Microsoft.ML.Trainers.MatrixFactorizationTrainer;

namespace MLNet.AutoPipeline.Test
{

    public class MockEstimatorBuilder : ISweepablePipelineNode
    {
        public MockEstimatorBuilder(string name)
        {
            this.EstimatorName = name;
        }

        public TransformerScope Scope => TransformerScope.Everything;

        public string EstimatorName { get; private set; }

        public IValueGenerator[] ValueGenerators => new List<IValueGenerator>().ToArray();

        public SweepablePipelineNodeType NodeType => SweepablePipelineNodeType.Sweepable;

        public IEstimator<ITransformer> BuildEstimator(ParameterSet parameters)
        {
            return null;
        }

        public string Summary()
        {
            throw new System.NotImplementedException();
        }
    }
}
