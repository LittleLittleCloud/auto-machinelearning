// <copyright file="FastTreeBinaryTrainerSweepableOptions.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers.FastTree;

namespace MLNet.AutoPipeline
{
    public class FastTreeBinaryTrainerSweepableOptions : SweepableOption<FastTreeBinaryTrainer.Options>
    {
        public static FastTreeBinaryTrainerSweepableOptions Default = new FastTreeBinaryTrainerSweepableOptions();

        [Parameter(nameof(FastTreeBinaryTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = CreateDiscreteParameter("Label");

        [Parameter(nameof(FastTreeBinaryTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = CreateDiscreteParameter("Features");

        [Parameter(nameof(FastTreeBinaryTrainer.Options.NumberOfLeaves))]
        public Parameter<int> NumberOfLeaves = CreateInt32Parameter(10, 1000, true);

        [Parameter(nameof(FastTreeBinaryTrainer.Options.NumberOfTrees))]
        public Parameter<int> NumberOfTrees = CreateInt32Parameter(1, 1000, true);

        [Parameter(nameof(FastTreeBinaryTrainer.Options.MinimumExampleCountPerLeaf))]
        public Parameter<int> MinimumExampleCountPerLeaf = CreateInt32Parameter(1, 100, true);

        [Parameter(nameof(FastTreeBinaryTrainer.Options.LearningRate))]
        public Parameter<double> LearningRate = CreateDoubleParameter(1e-4, 1, true);
    }
}
