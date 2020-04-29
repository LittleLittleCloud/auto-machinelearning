// <copyright file="SweepablePipelineTest.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading;
using FluentAssertions;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.Recommender;
using MLNet.AutoPipeline;
using MLNet.AutoPipeline.Test;
using MLNet.Sweeper;
using MLNet.Sweeper.Sweeper;
using Xunit;
using Xunit.Abstractions;
using static Microsoft.ML.Trainers.MatrixFactorizationTrainer;

namespace MLNet.AutoPipeline.Test
{
    public class SweepablePipelineTest
    {
        private ITestOutputHelper _output;

        public SweepablePipelineTest(ITestOutputHelper output)
        {
            this._output = output;
        }

        [Fact]
        public void SweepablePipeline_summary_should_work()
        {
            var singleNodeChain = new SweepablePipeline()
                                  .Append(new MockTransformer())
                                  .Append(new MockEstimatorBuilder("mockEstimator"));

            singleNodeChain.Summary().Should().Be("SweepablePipeline(MockTransformer=>mockEstimator)");
        }
    }
}
