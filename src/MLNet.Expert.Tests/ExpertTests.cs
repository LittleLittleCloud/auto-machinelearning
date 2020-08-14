// <copyright file="ExpertTests.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using FluentAssertions;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.AutoPipeline;
using MLNet.AutoPipeline.Metric;
using MLNet.Expert.AutoML;
using System;
using Xunit;
using Xunit.Abstractions;

namespace MLNet.Expert.Tests
{
    public class ExpertTests : TestBase
    {
        public ExpertTests(ITestOutputHelper helper)
            : base(helper)
        {
        }

        [Fact]
        public void NumericFeatureExpert_should_propose_NormalizeMeanVariance_and_NormalizeMinMax()
        {
            var context = new MLContext();
            var option = new NumericFeatureExpert.Option();

            var expert = new NumericFeatureExpert(context, option);
            expert.Propose("test").ToString()
                  .Should().Contain("SingleNode(NormalizeMeanVariance)")
                  .And
                  .Contain("SingleNode(NormalizeMinMax)");
        }

        [Fact]
        public void ClassificationExpert_should_propose_naiveBayes()
        {
            var expert = this.GetClassificationExpert();
            expert.Propose("label", "feature").ToString()
                  .Should().Contain("SingleNode(NaiveBayesMulticlassTrainer)");
        }

        [Fact]
        public void ClassificationExpert_should_propose_lbfgsMaximumEntropy()
        {
            var expert = this.GetClassificationExpert();
            expert.Propose("label", "feature").ToString()
                  .Should().Contain("SingleNode(LbfgsMaximumEntropyMulticlassTrainer)");
        }

        [Fact]
        public void ClassificationExpert_should_propose_lightGBM()
        {
            var expert = this.GetClassificationExpert();
            expert.Propose("label", "feature").ToString()
                  .Should().Contain("SingleNode(LightGbmMulticlassTrainer)");
        }

        [Fact]
        public async void AutoMLTestAsync()
        {
            var context = new MLContext();
            var classificationExpertOption = new ClassificationExpert.Option()
            {
            };

            var option = new Classification.Option()
            {
                BeamSearch = 3,
                ScoreMetric = new MicroAccuracyMetric(),
                LabelColumn = "species",
                MaximumTrainingTime = 60,
                ClassificationExpertOption = classificationExpertOption,
            };

            var reporter = new Reporter(this.Output);

            var dataset = context.Data.LoadFromTextFile<Iris>(this.GetFileFromTestData("iris.csv"), separatorChar: ',', hasHeader: true);
            var traintestSplit = context.Data.TrainTestSplit(dataset, 0.3);
            var trainValidateSplit = context.Data.TrainTestSplit(traintestSplit.TrainSet, 0.3);
            var experiment = new Classification(context, option);

            var result = await experiment.TrainAsync(trainValidateSplit.TrainSet, trainValidateSplit.TestSet, reporter);

            // test
            var testResult = result.BestModel.Transform(traintestSplit.TestSet);
            var score = context.MulticlassClassification.Evaluate(testResult, "species");
            this.Output.WriteLine($"test micro accuracy metric: {score.MicroAccuracy}");
        }

        private ClassificationExpert GetClassificationExpert()
        {
            var context = new MLContext();
            var expert = new ClassificationExpert(context, new ClassificationExpert.Option());
            return expert;
        }

        private class Iris
        {
            [LoadColumn(0)]
            public float sepal_length;

            [LoadColumn(1)]
            public float sepal_width;

            [LoadColumn(2)]
            public float petal_length;

            [LoadColumn(3)]
            public float petal_width;

            [LoadColumn(4)]
            public string species;
        }

        private class Reporter : IProgress<IterationInfo>
        {
            private ITestOutputHelper output;

            public Reporter(ITestOutputHelper output)
            {
                this.output = output;
            }

            public void Report(IterationInfo value)
            {
                this.output.WriteLine(value.SweepablePipeline.Summary());
                this.output.WriteLine(value.ParameterSet?.ToString() ?? string.Empty);
                this.output.WriteLine($"validate score: {value.ScoreMetric.Name}: {value.ScoreMetric.Score}");
                this.output.WriteLine($"training time: {value.TrainingTime}");
            }
        }
    }
}
