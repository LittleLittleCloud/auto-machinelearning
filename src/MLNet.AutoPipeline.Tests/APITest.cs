// <copyright file="APITest.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using FluentAssertions;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.LightGbm;
using MLNet.AutoPipeline.API.OptionBuilder;
using MLNet.Sweeper;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Xunit;
using Xunit.Abstractions;

namespace MLNet.AutoPipeline.Test
{
    public class APITest : TestBase
    {
        public APITest(ITestOutputHelper helper)
            : base(helper)
        {
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoPipeline_should_create_naive_bayes_classifier()
        {
            var context = new MLContext();
            var trainer = context.AutoML().MultiClassification.NaiveBayes("label", "feature");
            Approvals.Verify(trainer.ToCodeGenNodeContract());
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoPipeline_should_create_sdca_maximum_entropy_classifier_with_default_option()
        {
            var context = new MLContext();
            var trainer = context.AutoML().MultiClassification.SdcaMaximumEntropy("label", "feature");
            var parameterValues = SdcaMaximumEntropyOptionBuilder.Default.ValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new ParameterSet(parameterValues);
            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterset));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoPipeline_should_create_sdca_maximum_entropy_classifier_with_custom_option()
        {
            var context = new MLContext();
            var option = new CustomSdcaMaximumEntropyOptionBuilder();
            var trainer = context.AutoML().MultiClassification.SdcaMaximumEntropy("label", "feature", option);
            var parameterValues = option.ValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new ParameterSet(parameterValues);
            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterset));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoPipeline_should_create_sdca_non_calibrated_classifier_with_default_option()
        {
            var context = new MLContext();
            var trainer = context.AutoML().MultiClassification.SdcaNonCalibreated("label", "feature");
            var parameterValues = SdcaNonCalibratedOptionBuilder.Default.ValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new ParameterSet(parameterValues);
            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterset));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoPipeline_should_create_lbfgs_maximum_entropy_classifier_with_default_option()
        {
            var context = new MLContext();
            var trainer = context.AutoML().MultiClassification.LbfgsMaximumEntropy("label", "feature");
            var parameterValues = LbfgsMaximumEntropyOptionBuilder.Default.ValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new ParameterSet(parameterValues);
            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterset));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoPipeline_should_create_lightGbm_classifier_with_default_option()
        {
            var context = new MLContext();
            var trainer = context.AutoML().MultiClassification.LightGbm("label", "feature");
            var parameterValues = LightGbmOptionBuilder.Default.ValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new ParameterSet(parameterValues);
            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterset));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoPipeline_should_create_classifier_with_custom_factory()
        {
            var context = new MLContext();
            var optionBuilder = new CustomSdcaMaximumEntropyOptionBuilder();
            var trainer = context.AutoML().SweepableTrainer(
                                (context, option) =>
                                {
                                    option.LabelColumnName = "Label";
                                    option.FeatureColumnName = "Features";
                                    return context.MulticlassClassification.Trainers.SdcaMaximumEntropy(option);
                                },
                                optionBuilder,
                                new string[] { "Features" },
                                "Score",
                                "CustomSdca");

            var parameterValues = optionBuilder.ValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new ParameterSet(parameterValues);
            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterset));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoPipeline_should_create_ova_classifier_from_binary_classifier()
        {
            var context = new MLContext();
            var optionBuilder = new CustomSdcaMaximumEntropyOptionBuilder();
            var binaryTrainer = context.AutoML().SweepableTrainer(
                                (context, option) =>
                                {
                                    option.LabelColumnName = "Label";
                                    option.FeatureColumnName = "Features";
                                    return context.BinaryClassification.Trainers.SdcaLogisticRegression(option.LabelColumnName, option.FeatureColumnName);
                                },
                                optionBuilder,
                                new string[] { "Features" },
                                "Score",
                                "CustomSdca");
            var ovaTrainer = context.AutoML().MultiClassification.OneVersusAll(binaryTrainer);

            var parameterValues = optionBuilder.ValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new ParameterSet(parameterValues);
            Approvals.Verify(ovaTrainer.ToCodeGenNodeContract(parameterset));
        }

        [Fact]
        public void AutoPipeline_should_create_sweepable_pipeline_from_estimator_chain()
        {
            var context = new MLContext();
            var pipeline = context.Transforms.Conversion.MapValueToKey("species", "species")
                                  .Append(context.Transforms.Concatenate("features", new string[] { "sepal_length", "sepal_width", "petal_length", "petal_width" }))
                                  .Append(context.AutoML().MultiClassification.LightGbm("species", "features"));

            var parameterValues = pipeline.ValueGenerators.Select(x => x.Name);
            pipeline.Nodes.Count.Should().Be(3);
            parameterValues.Should().Equal(new string[] { "LearningRate", "NumberOfLeaves", "NumberOfIterations", "MinimumExampleCountPerLeaf" });
        }

        [Fact]
        public void AutoPipeline_should_create_sweepable_pipeline_from_estimator()
        {
            var context = new MLContext();
            var pipeline = context.Transforms.Conversion.MapValueToKey("species", "species")
                                  .Append(context.AutoML().MultiClassification.LightGbm("species", "features"));

            var parameterValues = pipeline.ValueGenerators.Select(x => x.Name);
            pipeline.Nodes.Count.Should().Be(2);
            parameterValues.Should().Equal(new string[] { "LearningRate", "NumberOfLeaves", "NumberOfIterations", "MinimumExampleCountPerLeaf" });
        }
    }

    public class CustomSdcaMaximumEntropyOptionBuilder : OptionBuilder<SdcaMaximumEntropyMulticlassTrainer.Options>
    {
        [SweepableParameter(1E-4F, 20f, true, 20)]
        public float L2Regularization;

        [Parameter]
        public float L1Regularization = 0.3f;

        [SweepableParameter(1E-4F, 10f, true, 20)]
        public float BiasLearningRate;
    }
}
