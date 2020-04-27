// <copyright file="SweepablePipelineTest.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Text;
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

        [Fact]
        public void SweepablePipeline_RecommendationE2ETest_RandomSweeper()
        {
            var context = new MLContext();
            var paramaters = new MFOption();
            var dataset = context.Data.LoadFromTextFile<ModelInput>(@".\TestData\recommendation-ratings-train.csv", separatorChar: ',', hasHeader: true);
            var split = context.Data.TrainTestSplit(dataset, 0.3);
            var sweeperOption = new UniformRandomSweeper.Option();

            var randomSweeper = new UniformRandomSweeper(sweeperOption);
            var pipelines = new SweepablePipeline()
                          .Append(context.Transforms.Conversion.MapValueToKey("userId", "userId"))
                          .Append(context.Transforms.Conversion.MapValueToKey("movieId", "movieId"))
                          .Append(context.Recommendation().Trainers.MatrixFactorization, paramaters, Microsoft.ML.Data.TransformerScope.Everything)
                          .Append(context.Transforms.CopyColumns("output", "Score"));

            pipelines.UseSweeper(randomSweeper);
            this._output.WriteLine(pipelines.Summary());
            foreach (var pipeline in pipelines.Sweeping(100))
            {
                var eval = pipeline.Fit(split.TrainSet).Transform(split.TestSet);
                var metrics = context.Regression.Evaluate(eval, "rating", "Score");
                this._output.WriteLine(randomSweeper.Current.ToString());
                this._output.WriteLine($"RMSE: {metrics.RootMeanSquaredError}");
            }
        }

        [Fact]
        public void SweepablePipeline_RecommendationE2ETest_GPSweeper()
        {
            var context = new MLContext();
            var paramaters = new MFOption();
            var dataset = context.Data.LoadFromTextFile<ModelInput>(@".\TestData\recommendation-ratings-train.csv", separatorChar: ',', hasHeader: true);
            var split = context.Data.TrainTestSplit(dataset, 0.3);

            var gpSweeper = new GaussProcessSweeper(new GaussProcessSweeper.Option());

            var mfTrainer = new SweepableNode<MatrixFactorizationPredictionTransformer, Options>(context.Recommendation().Trainers.MatrixFactorization, paramaters);

            var pipelines = new SweepablePipeline()
                           .Append(context.Transforms.Conversion.MapValueToKey("userId", "userId"))
                           .Append(context.Transforms.Conversion.MapValueToKey("movieId", "movieId"))
                           .Append(mfTrainer)
                           .Append(context.Transforms.CopyColumns("output", "Score"));

            pipelines.UseSweeper(gpSweeper);
            this._output.WriteLine(pipelines.Summary());

            foreach (var pipeline in pipelines.Sweeping(200))
            {
                var eval = pipeline.Fit(split.TrainSet).Transform(split.TestSet);
                var metrics = context.Regression.Evaluate(eval, "rating", "Score");
                this._output.WriteLine(gpSweeper.Current.ToString());
                var result = new RunResult(gpSweeper.Current, metrics.RootMeanSquaredError, true);
                gpSweeper.AddRunHistory(result);
                this._output.WriteLine($"RMSE: {metrics.RootMeanSquaredError}");
            }
        }

        private class MFOption : OptionBuilder<MatrixFactorizationTrainer.Options>
        {
            public string MatrixColumnIndexColumnName = "userId";

            public string MatrixRowIndexColumnName = "movieId";

            public string LabelColumnName = "rating";

            [Parameter("Alpha", 0.0001f, 1f, true)]
            public float Alpha = 0.0001f;

            [Parameter("ApproximationRank", 8, 128, steps: 20)]
            public int ApproximationRank = 8;

            [Parameter("Lambda", 0.01f, 1f, true, 20)]
            public double Lambda = 0.01f;

            [Parameter("LearningRate", 0.001f, 0.1f, true, 100)]
            public double LearningRate = 0.001f;

            [Parameter("LossFunctionType", new object[] { LossFunctionType.SquareLossOneClass, LossFunctionType.SquareLossRegression })]
            public LossFunctionType LossFunction;
        }

        private class ModelInput
        {
            [ColumnName("userId"), LoadColumn(0)]
            public float UserId { get; set; }

            [ColumnName("movieId"), LoadColumn(1)]
            public float MovieId { get; set; }

            [ColumnName("rating"), LoadColumn(2)]
            public float Rating { get; set; }
        }
    }
}
