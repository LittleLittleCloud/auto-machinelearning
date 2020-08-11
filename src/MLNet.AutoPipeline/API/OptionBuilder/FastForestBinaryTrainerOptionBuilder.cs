// <copyright file="FastForestBinaryTrainerOptionBuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers.FastTree;

namespace MLNet.AutoPipeline
{
    public class FastForestBinaryTrainerOptionBuilder : OptionBuilder<FastForestBinaryTrainer.Options>
    {
        public static FastForestBinaryTrainerOptionBuilder Default = new FastForestBinaryTrainerOptionBuilder();

        [Parameter]
        public Parameter<string> LabelColumnName = ParameterBuilder.CreateFromSingleValue("Label");

        [Parameter]
        public Parameter<string> FeatureColumnName = ParameterBuilder.CreateFromSingleValue("Features");

        [Parameter]
        public Parameter<string> ExampleWeightColumnName = ParameterBuilder.CreateFromSingleValue<string>(default);

        [Parameter]
        public Parameter<int> NumberOfLeaves = ParameterBuilder.CreateInt32Parameter(1, 1000, true);

        [Parameter]
        public Parameter<int> NumberOfTrees = ParameterBuilder.CreateInt32Parameter(1, 1000, true);

        [Parameter]
        public Parameter<int> MinimumExampleCountPerLeaf = ParameterBuilder.CreateInt32Parameter(1, 100, true);
    }
}
