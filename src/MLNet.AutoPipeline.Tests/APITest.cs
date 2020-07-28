// <copyright file="APITest.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using FluentAssertions;
using Microsoft.ML;
using Microsoft.ML.Trainers;
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
