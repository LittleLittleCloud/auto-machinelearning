// <copyright file="LightGbmBinaryTrainerSweepableOptions.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers.LightGbm;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    public class LightGbmBinaryTrainerSweepableOptions : SweepableOption<LightGbmBinaryTrainer.Options>
    {
        public static LightGbmBinaryTrainerSweepableOptions Default = new LightGbmBinaryTrainerSweepableOptions();

        [Parameter(nameof(LightGbmBinaryTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = CreateDiscreteParameter("Label");

        [Parameter(nameof(LightGbmBinaryTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = CreateDiscreteParameter("Features");

        [Parameter(nameof(LightGbmBinaryTrainer.Options.NumberOfLeaves))]
        public Parameter<int> NumberOfLeaves = CreateInt32Parameter(10, 1000, true);

        [Parameter(nameof(LightGbmBinaryTrainer.Options.MinimumExampleCountPerLeaf))]
        public Parameter<int> MinimumExampleCountPerLeaf = CreateInt32Parameter(1, 100, true);

        [Parameter(nameof(LightGbmBinaryTrainer.Options.LearningRate))]
        public Parameter<double> LearningRate = CreateDoubleParameter(1e-4, 1, true);
    }
}
