// <copyright file="LightGbmBinaryTrainerSweepableOptions.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers.LightGbm;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    public class LightGbmRegressionTrainerSweepableOptions : SweepableOption<LightGbmRegressionTrainer.Options>
    {
        public static LightGbmRegressionTrainerSweepableOptions Default = new LightGbmRegressionTrainerSweepableOptions();

        [Parameter(nameof(LightGbmRegressionTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = CreateDiscreteParameter("Label");

        [Parameter(nameof(LightGbmRegressionTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = CreateDiscreteParameter("Features");

        [Parameter(nameof(LightGbmRegressionTrainer.Options.NumberOfLeaves))]
        public Parameter<int> NumberOfLeaves = CreateInt32Parameter(10, 1000, true);

        [Parameter(nameof(LightGbmRegressionTrainer.Options.MinimumExampleCountPerLeaf))]
        public Parameter<int> MinimumExampleCountPerLeaf = CreateInt32Parameter(1, 100, true);

        [Parameter(nameof(LightGbmRegressionTrainer.Options.LearningRate))]
        public Parameter<double> LearningRate = CreateDoubleParameter(1e-4, 1, true);

        [Parameter(nameof(LightGbmRegressionTrainer.Options.NumberOfIterations))]
        public Parameter<int> NumberOfIteratioins = CreateInt32Parameter(20, 1000, true);
    }
}
