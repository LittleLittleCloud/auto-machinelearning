// <copyright file="FastTreeBinaryTrainerOptionBuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers.FastTree;

namespace MLNet.AutoPipeline
{
    public class FastTreeBinaryTrainerOptionBuilder : OptionBuilder<FastTreeBinaryTrainer.Options>
    {
        public static FastTreeBinaryTrainerOptionBuilder Default = new FastTreeBinaryTrainerOptionBuilder();

        [Parameter(nameof(FastTreeBinaryTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = ParameterBuilder.CreateFromSingleValue("Label");

        [Parameter(nameof(FastTreeBinaryTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = ParameterBuilder.CreateFromSingleValue("Features");

        [Parameter(nameof(FastTreeBinaryTrainer.Options.ExampleWeightColumnName))]
        public Parameter<string> ExampleWeightColumnName = ParameterBuilder.CreateFromSingleValue<string>(default);

        [Parameter(nameof(FastTreeBinaryTrainer.Options.NumberOfLeaves))]
        public Parameter<int> NumberOfLeaves = ParameterBuilder.CreateInt32Parameter(1, 1000, true);

        [Parameter(nameof(FastTreeBinaryTrainer.Options.NumberOfTrees))]
        public Parameter<int> NumberOfTrees = ParameterBuilder.CreateInt32Parameter(1, 1000, true);

        [Parameter(nameof(FastTreeBinaryTrainer.Options.MinimumExampleCountPerLeaf))]
        public Parameter<int> MinimumExampleCountPerLeaf = ParameterBuilder.CreateInt32Parameter(1, 100, true);

        [Parameter(nameof(FastTreeBinaryTrainer.Options.LearningRate))]
        public Parameter<double> LearningRate = ParameterBuilder.CreateDoubleParameter(1e-4, 1, true);


    }
}
