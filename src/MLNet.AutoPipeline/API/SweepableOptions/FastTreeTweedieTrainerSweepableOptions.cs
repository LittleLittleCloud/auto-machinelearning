// <copyright file="FastTreeTweedieTrainerSweepableOptions.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers.FastTree;

namespace MLNet.AutoPipeline
{
    public class FastTreeTweedieTrainerSweepableOptions : SweepableOption<FastTreeTweedieTrainer.Options>
    {
        public static FastTreeTweedieTrainerSweepableOptions Default = new FastTreeTweedieTrainerSweepableOptions();

        [Parameter(nameof(FastTreeTweedieTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = CreateDiscreteParameter("Label");

        [Parameter(nameof(FastTreeTweedieTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = CreateDiscreteParameter("Features");

        [Parameter(nameof(FastTreeTweedieTrainer.Options.NumberOfLeaves))]
        public Parameter<int> NumberOfLeaves = CreateInt32Parameter(10, 1000, true);

        [Parameter(nameof(FastTreeTweedieTrainer.Options.NumberOfTrees))]
        public Parameter<int> NumberOfTrees = CreateInt32Parameter(1, 1000, true);

        [Parameter(nameof(FastTreeTweedieTrainer.Options.MinimumExampleCountPerLeaf))]
        public Parameter<int> MinimumExampleCountPerLeaf = CreateInt32Parameter(1, 100, true);

        [Parameter(nameof(FastTreeTweedieTrainer.Options.LearningRate))]
        public Parameter<double> LearningRate = CreateDoubleParameter(1e-4, 1, true);
    }
}
