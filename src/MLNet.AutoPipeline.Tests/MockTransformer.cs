// <copyright file="MockTransformer.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using FluentAssertions;
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

    public class MockTransformer : IEstimator<ITransformer>
    {
        public ITransformer Fit(IDataView input)
        {
            throw new System.NotImplementedException();
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            throw new System.NotImplementedException();
        }
    }
}
