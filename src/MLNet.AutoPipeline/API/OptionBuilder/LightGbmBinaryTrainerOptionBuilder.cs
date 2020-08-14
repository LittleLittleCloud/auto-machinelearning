// <copyright file="LightGbmBinaryTrainerOptionBuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers.LightGbm;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    public class LightGbmBinaryTrainerOptionBuilder : SweepableOption<LightGbmBinaryTrainer.Options>
    {
        public static LightGbmBinaryTrainerOptionBuilder Default = new LightGbmBinaryTrainerOptionBuilder();

        [Parameter(nameof(LightGbmBinaryTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = ParameterBuilder.CreateFromSingleValue("Label");

        [Parameter(nameof(LightGbmBinaryTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = ParameterBuilder.CreateFromSingleValue("Features");

        [Parameter(nameof(LightGbmBinaryTrainer.Options.ExampleWeightColumnName))]
        public Parameter<string> ExampleWeightColumnName = ParameterBuilder.CreateFromSingleValue<string>(default);

        [Parameter(nameof(LightGbmBinaryTrainer.Options.NumberOfLeaves))]
        public Parameter<int> NumberOfLeaves = ParameterBuilder.CreateInt32Parameter(1, 1000, true);

        [Parameter(nameof(LightGbmBinaryTrainer.Options.MinimumExampleCountPerLeaf))]
        public Parameter<int> MinimumExampleCountPerLeaf = ParameterBuilder.CreateInt32Parameter(1, 100, true);

        [Parameter(nameof(LightGbmBinaryTrainer.Options.LearningRate))]
        public Parameter<double> LearningRate = ParameterBuilder.CreateDoubleParameter(1e-4, 1, true);
    }
}
