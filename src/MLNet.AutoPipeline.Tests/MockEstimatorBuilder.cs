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

    public class MockEstimatorBuilder : SweepableEstimatorBase
    {
        public MockEstimatorBuilder(string name)
            :base(name, null, null, null, TransformerScope.Everything)
        {
            this.EstimatorName = name;
        }

        public override IEstimator<ITransformer> BuildFromParameters(IDictionary<string, string> parameters)
        {
            throw new System.NotImplementedException();
        }
    }
}
