// <copyright file="FastForestRegressionTrainerSweepableOptions.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers.FastTree;

namespace MLNet.AutoPipeline
{
    public class FastForestRegressionTrainerSweepableOptions : SweepableOption<FastForestRegressionTrainer.Options>
    {
        public static FastForestRegressionTrainerSweepableOptions Default = new FastForestRegressionTrainerSweepableOptions();

        [Parameter(nameof(FastForestRegressionTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = CreateDiscreteParameter("Label");

        [Parameter(nameof(FastForestRegressionTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = CreateDiscreteParameter("Features");

        [Parameter(nameof(FastForestRegressionTrainer.Options.NumberOfLeaves))]
        public Parameter<int> NumberOfLeaves = CreateInt32Parameter(10, 1000, true);

        [Parameter(nameof(FastForestRegressionTrainer.Options.NumberOfTrees))]
        public Parameter<int> NumberOfTrees = CreateInt32Parameter(1, 1000, true);

        [Parameter(nameof(FastForestRegressionTrainer.Options.MinimumExampleCountPerLeaf))]
        public Parameter<int> MinimumExampleCountPerLeaf = CreateInt32Parameter(1, 100, true);
    }
}
