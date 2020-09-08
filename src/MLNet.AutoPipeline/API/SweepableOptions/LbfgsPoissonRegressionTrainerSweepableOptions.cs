// <copyright file="LbfgsPoissonRegressionTrainerSweepableOptions.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers;

namespace MLNet.AutoPipeline
{
    public class LbfgsPoissonRegressionTrainerSweepableOptions : SweepableOption<LbfgsPoissonRegressionTrainer.Options>
    {
        public static LbfgsPoissonRegressionTrainerSweepableOptions Default = new LbfgsPoissonRegressionTrainerSweepableOptions();

        [Parameter(nameof(LbfgsPoissonRegressionTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = CreateDiscreteParameter("Label");

        [Parameter(nameof(LbfgsPoissonRegressionTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = CreateDiscreteParameter("Features");

        [Parameter(nameof(LbfgsPoissonRegressionTrainer.Options.L1Regularization))]
        public Parameter<float> L1Regularization = CreateFloatParameter(1e-3f, 100f, true);

        [Parameter(nameof(LbfgsPoissonRegressionTrainer.Options.L2Regularization))]
        public Parameter<float> L2Regularization = CreateFloatParameter(1e-5f, 100f, true);

        [Parameter(nameof(LbfgsPoissonRegressionTrainer.Options.OptimizationTolerance))]
        public Parameter<float> OptimizationTolerance = CreateFloatParameter(1e-9f, 1e-4f, true);

        [Parameter(nameof(LbfgsPoissonRegressionTrainer.Options.HistorySize))]
        public Parameter<int> HistorySize = CreateInt32Parameter(5, 100, false);

        [Parameter(nameof(LbfgsPoissonRegressionTrainer.Options.EnforceNonNegativity))]
        public Parameter<bool> EnforceNonNegativity = CreateDiscreteParameter(false);
    }
}
