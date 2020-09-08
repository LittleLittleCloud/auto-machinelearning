// <copyright file="LbfgsLogisticRegressionBinaryTrainerSweepableOptions.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers;

namespace MLNet.AutoPipeline
{
    public class LbfgsLogisticRegressionBinaryTrainerSweepableOptions : SweepableOption<LbfgsLogisticRegressionBinaryTrainer.Options>
    {
        public static LbfgsLogisticRegressionBinaryTrainerSweepableOptions Default = new LbfgsLogisticRegressionBinaryTrainerSweepableOptions();

        [Parameter(nameof(LbfgsLogisticRegressionBinaryTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = CreateDiscreteParameter("Label");

        [Parameter(nameof(LbfgsLogisticRegressionBinaryTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = CreateDiscreteParameter("Features");

        [Parameter(nameof(LbfgsLogisticRegressionBinaryTrainer.Options.L1Regularization))]
        public Parameter<float> L1Regularization = CreateFloatParameter(1e-3f, 100f, true);

        [Parameter(nameof(LbfgsLogisticRegressionBinaryTrainer.Options.L2Regularization))]
        public Parameter<float> L2Regularization = CreateFloatParameter(1e-5f, 100f, true);

        [Parameter(nameof(LbfgsLogisticRegressionBinaryTrainer.Options.MaximumNumberOfIterations))]
        public Parameter<int> MaximumNumberOfIterations = CreateInt32Parameter(1, 256, true);

        [Parameter(nameof(LbfgsLogisticRegressionBinaryTrainer.Options.OptimizationTolerance))]
        public Parameter<float> OptimizationTolerance = CreateFloatParameter(1e-9f, 1e-4f, true);

        [Parameter(nameof(LbfgsLogisticRegressionBinaryTrainer.Options.HistorySize))]
        public Parameter<int> HistorySize = CreateInt32Parameter(5, 100, false);

        [Parameter(nameof(LbfgsLogisticRegressionBinaryTrainer.Options.EnforceNonNegativity))]
        public Parameter<bool> EnforceNonNegativity = CreateDiscreteParameter(false);
    }
}
