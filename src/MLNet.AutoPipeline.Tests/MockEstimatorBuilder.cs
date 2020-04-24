// <copyright file="MockEstimatorBuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using MLNet.AutoPipeline;
using MLNet.Sweeper;
using Xunit;
using Xunit.Abstractions;
using static Microsoft.ML.Trainers.MatrixFactorizationTrainer;

namespace MLNet.AutoPipeline.Test
{

    public class MockEstimatorBuilder : IEstimatorBuilder
    {
        public MockEstimatorBuilder(string name)
        {
            this.EstimatorName = name;
        }

        public TransformerScope Scope => TransformerScope.Everything;

        public string EstimatorName { get; private set; }

        public IEstimator<ITransformer> BuildEstimator(ParameterSet parameters)
        {
            return null;
        }
    }
}
