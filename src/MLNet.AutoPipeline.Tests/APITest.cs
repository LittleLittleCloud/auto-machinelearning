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
            var parameterValues = SdcaMaximumEntropyMulticlassTrainerSweepableOptions.Default.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new Parameters(parameterValues);
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
            var parameterValues = option.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new Parameters(parameterValues);
            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterset));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoPipeline_should_create_sdca_non_calibrated_classifier_with_default_option()
        {
            var context = new MLContext();
            var trainer = context.AutoML().MultiClassification.SdcaNonCalibreated("label", "feature");
            var parameterValues = SdcaNonCalibratedMulticlassTrainerSweepableOptions.Default.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new Parameters(parameterValues);
            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterset));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoPipeline_should_create_lbfgs_maximum_entropy_classifier_with_default_option()
        {
            var context = new MLContext();
            var trainer = context.AutoML().MultiClassification.LbfgsMaximumEntropy("label", "feature");
            var parameterValues = LbfgsMaximumEntropyMulticlassTrainerSweepableOptions.Default.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new Parameters(parameterValues);
            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterset));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoPipeline_should_create_lightGbm_classifier_with_default_option()
        {
            var context = new MLContext();
            var trainer = context.AutoML().MultiClassification.LightGbm("label", "feature");
            var parameterValues = LightGbmMulticlassTrainerSweepableOptions.Default.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new Parameters(parameterValues);
            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterset));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoPipeline_should_create_classifier_with_custom_factory()
        {
            var context = new MLContext();
            var optionBuilder = new CustomSdcaMaximumEntropyOptionBuilder();
            var trainer = context.AutoML().CreateSweepableEstimator(
                                (context, option) =>
                                {
                                    option.LabelColumnName = "Label";
                                    option.FeatureColumnName = "Features";
                                    return context.MulticlassClassification.Trainers.SdcaMaximumEntropy(option);
                                },
                                optionBuilder,
                                new string[] { "Features" },
                                new string[] { "Score" },
                                "CustomSdca");

            var parameterValues = optionBuilder.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new Parameters(parameterValues);
            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterset));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoPipeline_should_create_ova_classifier_from_binary_classifier()
        {
            var context = new MLContext();
            var optionBuilder = new CustomSdcaMaximumEntropyOptionBuilder();
            var binaryTrainer = context.AutoML().CreateSweepableEstimator(
                                (context, option) =>
                                {
                                    option.LabelColumnName = "Label";
                                    option.FeatureColumnName = "Features";
                                    return context.BinaryClassification.Trainers.SdcaLogisticRegression(option.LabelColumnName, option.FeatureColumnName);
                                },
                                optionBuilder,
                                new string[] { "Features" },
                                new string[] { "Score" },
                                "CustomSdca");
            var ovaTrainer = context.AutoML().MultiClassification.OneVersusAll(binaryTrainer);

            var parameterValues = optionBuilder.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new Parameters(parameterValues);
            Approvals.Verify(ovaTrainer.ToCodeGenNodeContract(parameterset));
        }

        [Fact]
        public void AutoPipeline_should_create_sweepable_pipeline_from_estimator_chain()
        {
            var context = new MLContext();
            var pipeline = context.Transforms.Conversion.MapValueToKey("species", "species")
                                  .Append(context.Transforms.Concatenate("features", new string[] { "sepal_length", "sepal_width", "petal_length", "petal_width" }))
                                  .Append(context.AutoML().MultiClassification.LightGbm("species", "features"));

            var parameterValues = pipeline.EstimatorGenerators[2].Values[0].SweepableValueGenerators.Select(x => x.Name);
            pipeline.EstimatorGenerators.Count.Should().Be(3);
            parameterValues.Should().Equal(new string[] { "LearningRate", "NumberOfLeaves", "NumberOfIterations", "MinimumExampleCountPerLeaf" });
        }

        [Fact]
        public void AutoPipeline_should_create_sweepable_pipeline_from_estimator()
        {
            var context = new MLContext();
            var pipeline = context.Transforms.Conversion.MapValueToKey("species", "species")
                                  .Append(context.AutoML().MultiClassification.LightGbm("species", "features"));

            var parameterValues = pipeline.EstimatorGenerators[1].Values[0].SweepableValueGenerators.Select(x => x.Name);
            pipeline.EstimatorGenerators.Count.Should().Be(2);
            parameterValues.Should().Equal(new string[] { "LearningRate", "NumberOfLeaves", "NumberOfIterations", "MinimumExampleCountPerLeaf" });
        }

        [Fact]
        public void AutoML_should_create_sweepable_pipeline_from_INode_using_extension()
        {
            var context = new MLContext();
            var pipeline = context.AutoML().CreateUnsweepableEstimator(context.Transforms.Conversion.MapKeyToValue("species", "species"))
                                  .Append(context.AutoML().MultiClassification.LightGbm("species", "features"));

            pipeline.ToString().Should().Be("SweepablePipeline([KeyToValueMappingEstimator]=>[LightGbmMulticlassTrainer])");

            pipeline = context.AutoML().MultiClassification.LightGbm("species", "features")
                                  .Append(context.AutoML().CreateUnsweepableEstimator(context.Transforms.Conversion.MapKeyToValue("species", "species")));

            pipeline.ToString().Should().Be("SweepablePipeline([LightGbmMulticlassTrainer]=>[KeyToValueMappingEstimator])");

            pipeline = context.AutoML().MultiClassification.LightGbm("species", "features")
                      .Append(context.Transforms.Conversion.MapKeyToValue("species", "species"));

            pipeline.ToString().Should().Be("SweepablePipeline([LightGbmMulticlassTrainer]=>[KeyToValueMappingEstimator])");
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoML_should_create_linear_svm_classifier_with_option()
        {
            var context = new MLContext();
            var optionBuilder = LinearSvmBinaryTrainerSweepableOptions.Default;
            var trainer = context.AutoML().BinaryClassification.LinearSvm("label", "feature", optionBuilder);
            var parameterValue = optionBuilder.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new Parameters(parameterValue);
            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterset));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoML_should_create_fast_forest_binary_classifier_with_option()
        {
            var context = new MLContext();
            var optionBuilder = FastForestBinaryTrainerSweepableOptions.Default;
            var trainer = context.AutoML().BinaryClassification.FastForest("label", "feature", optionBuilder);
            var parameterValue = optionBuilder.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new Parameters(parameterValue);
            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterset));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoML_should_create_fast_tree_binary_classifier_with_option()
        {
            var context = new MLContext();
            var optionBuilder = FastTreeBinaryTrainerSweepableOptions.Default;
            var trainer = context.AutoML().BinaryClassification.FastTree("label", "feature", optionBuilder);
            var parameterValue = optionBuilder.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new Parameters(parameterValue);
            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterset));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoML_should_create_lightGbm_binary_classifier_with_option()
        {
            var context = new MLContext();
            var optionBuilder = LightGbmBinaryTrainerSweepableOptions.Default;
            var trainer = context.AutoML().BinaryClassification.LightGbm("label", "feature", optionBuilder);
            var parameterValue = optionBuilder.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new Parameters(parameterValue);
            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterset));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoML_should_create_ld_svm_classifier_with_option()
        {
            var context = new MLContext();
            var optionBuilder = LdSvmBinaryTrainerSweepableOptions.Default;
            var trainer = context.AutoML().BinaryClassification.LdSvm("label", "feature", optionBuilder);
            var parameterValue = optionBuilder.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new Parameters(parameterValue);
            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterset));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoML_should_create_gam_classifier_with_option()
        {
            var context = new MLContext();
            var optionBuilder = GamBinaryTrainerSweepableOptions.Default;
            var trainer = context.AutoML().BinaryClassification.Gam("label", "feature", optionBuilder);
            var parameterValue = optionBuilder.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new Parameters(parameterValue);
            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterset));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoML_should_create_sgd_noncalibrated_classifier_with_option()
        {
            var context = new MLContext();
            var optionBuilder = SgdNonCalibratedBinaryTrainerSweepableOptions.Default;
            var trainer = context.AutoML().BinaryClassification.SgdNonCalibrated("label", "feature", optionBuilder);
            var parameterValue = optionBuilder.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new Parameters(parameterValue);
            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterset));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoML_should_create_sgd_calibrated_classifier_with_option()
        {
            var context = new MLContext();
            var optionBuilder = SgdCalibratedBinaryTrainerSweepableOptions.Default;
            var trainer = context.AutoML().BinaryClassification.SgdCalibrated("label", "feature", optionBuilder);
            var parameterValue = optionBuilder.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new Parameters(parameterValue);
            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterset));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoML_should_create_sdca_non_calibrated_classifier_with_option()
        {
            var context = new MLContext();
            var optionBuilder = SdcaNonCalibratedBinaryTrainerSweepableOptions.Default;
            var trainer = context.AutoML().BinaryClassification.SdcaNonCalibrated("label", "feature", optionBuilder);
            var parameterValue = optionBuilder.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new Parameters(parameterValue);
            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterset));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoML_should_create_sdca_logistic_regression_binary_classifier_with_option()
        {
            var context = new MLContext();
            var optionBuilder = SdcaLogisticRegressionBinaryTrainerSweepableOptions.Default;
            var trainer = context.AutoML().BinaryClassification.SdcaLogisticRegression("label", "feature", optionBuilder);
            var parameterValue = optionBuilder.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new Parameters(parameterValue);
            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterset));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoML_should_create_lbfgs_logistic_regression_binary_classifier_with_option()
        {
            var context = new MLContext();
            var optionBuilder = LbfgsLogisticRegressionBinaryTrainerSweepableOptions.Default;
            var trainer = context.AutoML().BinaryClassification.LbfgsLogisticRegression("label", "feature", optionBuilder);
            var parameterValue = optionBuilder.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new Parameters(parameterValue);
            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterset));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoML_should_create_averaged_perceptron_binary_classifier_with_option()
        {
            var context = new MLContext();
            var optionBuilder = AveragedPerceptronBinaryTrainerSweepableOptions.Default;
            var trainer = context.AutoML().BinaryClassification.AveragedPerceptron("label", "feature", optionBuilder);
            var parameterValue = optionBuilder.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterset = new Parameters(parameterValue);
            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterset));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoML_should_create_light_gbm_regressor_with_default_option()
        {
            var context = new MLContext();
            context.Log += this.Context_Log;
            var optionSweeper = LightGbmRegressionTrainerSweepableOptions.Default;
            var trainer = context.AutoML().Regression.LightGbm();
            var parameterValue = optionSweeper.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterSet = new Parameters(parameterValue);

            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterSet));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoML_should_create_lbfgs_poisson_regressor_with_default_option()
        {
            var context = new MLContext();
            context.Log += this.Context_Log;
            var optionSweeper = LbfgsPoissonRegressionTrainerSweepableOptions.Default;
            var trainer = context.AutoML().Regression.LbfgsPoissonRegression();
            var parameterValue = optionSweeper.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterSet = new Parameters(parameterValue);

            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterSet));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoML_should_create_online_gradient_descent_regressor_with_default_option()
        {
            var context = new MLContext();
            context.Log += this.Context_Log;
            var optionSweeper = OnlineGradientDescentTrainerSweepableOptions.Default;
            var trainer = context.AutoML().Regression.OnlineGradientDescent();
            var parameterValue = optionSweeper.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterSet = new Parameters(parameterValue);

            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterSet));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoML_should_create_sdca_regressor_with_default_option()
        {
            var context = new MLContext();
            context.Log += this.Context_Log;
            var optionSweeper = SdcaRegressionTrainerSweepableOptions.Default;
            var trainer = context.AutoML().Regression.Sdca();
            var parameterValue = optionSweeper.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterSet = new Parameters(parameterValue);

            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterSet));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoML_should_create_fast_forest_regressor_with_default_option()
        {
            var context = new MLContext();
            context.Log += this.Context_Log;
            var optionSweeper = FastForestRegressionTrainerSweepableOptions.Default;
            var trainer = context.AutoML().Regression.FastForest();
            var parameterValue = optionSweeper.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterSet = new Parameters(parameterValue);

            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterSet));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoML_should_create_fast_tree_regressor_with_default_option()
        {
            var context = new MLContext();
            context.Log += this.Context_Log;
            var optionSweeper = FastTreeRegressionTrainerSweepableOptions.Default;
            var trainer = context.AutoML().Regression.FastTree();
            var parameterValue = optionSweeper.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterSet = new Parameters(parameterValue);

            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterSet));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoML_should_create_fast_tree_tweedie_regressor_with_default_option()
        {
            var context = new MLContext();
            context.Log += this.Context_Log;
            var optionSweeper = FastTreeTweedieTrainerSweepableOptions.Default;
            var trainer = context.AutoML().Regression.FastTreeTweedie();
            var parameterValue = optionSweeper.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterSet = new Parameters(parameterValue);

            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterSet));
        }

        [Fact]
        [UseApprovalSubdirectory("ApprovalTests")]
        [UseReporter(typeof(DiffReporter))]
        public void AutoML_should_create_gam_regressor_with_default_option()
        {
            var context = new MLContext();
            context.Log += this.Context_Log;
            var optionSweeper = GamRegressionTrainerSweepableOptions.Default;
            var trainer = context.AutoML().Regression.Gam();
            var parameterValue = optionSweeper.SweepableValueGenerators.Select(x => x.CreateFromNormalized(0.5));
            var parameterSet = new Parameters(parameterValue);

            Approvals.Verify(trainer.ToCodeGenNodeContract(parameterSet));
        }

        [Fact]
        public void AutoMLCatalog_should_only_create_once()
        {
            var context = new MLContext();
            context.Log += this.Context_Log;
            var autoMLCatalog = context.AutoML();
            var autoMLCatalog2 = context.AutoML();
            autoMLCatalog.Should().Equals(autoMLCatalog2);
        }

        private void Context_Log(object sender, LoggingEventArgs e)
        {
            this.Output.WriteLine(e.Message);
        }
    }

    public class CustomSdcaMaximumEntropyOptionBuilder : SweepableOption<SdcaMaximumEntropyMulticlassTrainer.Options>
    {
        [Parameter]
        public Parameter<float> L2Regularization = ParameterFactory.CreateFloatParameter(1E-4F, 20f, true, 20);

        [Parameter]
        public Parameter<float> L1Regularization = ParameterFactory.CreateDiscreteParameter(0.3f);

        [Parameter]
        public Parameter<float> BiasLearningRate = ParameterFactory.CreateFloatParameter(1E-4F, 10f, true, 20);
    }
}
