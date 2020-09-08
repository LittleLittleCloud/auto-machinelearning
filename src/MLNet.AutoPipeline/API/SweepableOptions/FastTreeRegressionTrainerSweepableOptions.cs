// <copyright file="FastTreeRegressionTrainerSweepableOptions.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers.FastTree;

namespace MLNet.AutoPipeline
{
    public class FastTreeRegressionTrainerSweepableOptions : SweepableOption<FastTreeRegressionTrainer.Options>
    {
        public static FastTreeRegressionTrainerSweepableOptions Default = new FastTreeRegressionTrainerSweepableOptions();

        [Parameter(nameof(FastTreeRegressionTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = CreateDiscreteParameter("Label");

        [Parameter(nameof(FastTreeRegressionTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = CreateDiscreteParameter("Features");

        [Parameter(nameof(FastTreeRegressionTrainer.Options.NumberOfLeaves))]
        public Parameter<int> NumberOfLeaves = CreateInt32Parameter(10, 1000, true);

        [Parameter(nameof(FastTreeRegressionTrainer.Options.NumberOfTrees))]
        public Parameter<int> NumberOfTrees = CreateInt32Parameter(1, 1000, true);

        [Parameter(nameof(FastTreeRegressionTrainer.Options.MinimumExampleCountPerLeaf))]
        public Parameter<int> MinimumExampleCountPerLeaf = CreateInt32Parameter(1, 100, true);

        [Parameter(nameof(FastTreeRegressionTrainer.Options.LearningRate))]
        public Parameter<double> LearningRate = CreateDoubleParameter(1e-4, 1, true);
    }
}
