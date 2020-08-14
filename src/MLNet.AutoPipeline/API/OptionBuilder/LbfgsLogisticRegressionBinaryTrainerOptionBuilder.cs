// <copyright file="LbfgsLogisticRegressionBinaryTrainerOptionBuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers;

namespace MLNet.AutoPipeline
{
    public class LbfgsLogisticRegressionBinaryTrainerOptionBuilder : SweepableOption<LbfgsLogisticRegressionBinaryTrainer.Options>
    {
        public static LbfgsLogisticRegressionBinaryTrainerOptionBuilder Default = new LbfgsLogisticRegressionBinaryTrainerOptionBuilder();

        [Parameter(nameof(LbfgsLogisticRegressionBinaryTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = ParameterBuilder.CreateFromSingleValue("Label");

        [Parameter(nameof(LbfgsLogisticRegressionBinaryTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = ParameterBuilder.CreateFromSingleValue("Features");

        [Parameter(nameof(LbfgsLogisticRegressionBinaryTrainer.Options.ExampleWeightColumnName))]
        public Parameter<string> ExampleWeightColumnName = ParameterBuilder.CreateFromSingleValue<string>(default);

        [Parameter(nameof(LbfgsLogisticRegressionBinaryTrainer.Options.L1Regularization))]
        public Parameter<float> L1Regularization = ParameterBuilder.CreateFloatParameter(1e-3f, 100f, true);

        [Parameter(nameof(LbfgsLogisticRegressionBinaryTrainer.Options.L2Regularization))]
        public Parameter<float> L2Regularization = ParameterBuilder.CreateFloatParameter(1e-5f, 100f, true);

        [Parameter(nameof(LbfgsLogisticRegressionBinaryTrainer.Options.MaximumNumberOfIterations))]
        public Parameter<int> MaximumNumberOfIterations = ParameterBuilder.CreateInt32Parameter(1, 256, true);

        [Parameter(nameof(LbfgsLogisticRegressionBinaryTrainer.Options.OptimizationTolerance))]
        public Parameter<float> OptimizationTolerance = ParameterBuilder.CreateFloatParameter(1e-9f, 1e-4f, true);

        [Parameter(nameof(LbfgsLogisticRegressionBinaryTrainer.Options.HistorySize))]
        public Parameter<int> HistorySize = ParameterBuilder.CreateInt32Parameter(5, 100, false);

        [Parameter(nameof(LbfgsLogisticRegressionBinaryTrainer.Options.EnforceNonNegativity))]
        public Parameter<bool> EnforceNonNegativity = ParameterBuilder.CreateFromSingleValue(false);
    }
}
