// <copyright file="SweepablePipelineTest.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using System.Threading;
using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using FluentAssertions;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.Recommender;
using MLNet.AutoPipeline;
using MLNet.AutoPipeline.Test;
using MLNet.Sweeper;
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
            var sweepablePipeline = new SweepablePipeline()
                                  .Append(new MockTransformer())
                                  .Append(new MockEstimatorBuilder("mockEstimator"));

            sweepablePipeline.ToString().Should().Be("SweepablePipeline([MockTransformer]=>[mockEstimator])");
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void SweepablePipeline_should_create_SingleEstimatorSweepablePipeline()
        {
            var sweepablePipeline = new SweepablePipeline()
                                    .Append(new MockEstimatorBuilder("estimator1_1"))
                                    .Append(new MockEstimatorBuilder("estimator2_1"), new MockEstimatorBuilder("estimator2_2"))
                                    .Append(new MockEstimatorBuilder("estimator3_1"), new MockEstimatorBuilder("estimator3_2"), new MockEstimatorBuilder("estimator3_3"));

            var res = new StringBuilder();
            res.AppendLine(sweepablePipeline.ToString());

            var sweeper = new GridSearchSweeper();
            foreach (var param in sweeper.ProposeSweeps(sweepablePipeline, 100))
            {
                var singleEstimatorPipeline = sweepablePipeline.BuildFromParameters(param);
                res.AppendLine(singleEstimatorPipeline.ToString());
            }

            Approvals.Verify(res.ToString());
        }
    }
}
