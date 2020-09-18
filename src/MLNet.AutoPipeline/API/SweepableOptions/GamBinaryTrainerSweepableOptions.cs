// <copyright file="GamBinaryTrainerSweepableOptions.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers.FastTree;

namespace MLNet.AutoPipeline
{
    public class GamBinaryTrainerSweepableOptions : SweepableOption<GamBinaryTrainer.Options>
    {
        public static GamBinaryTrainerSweepableOptions Default = new GamBinaryTrainerSweepableOptions();

        [Parameter(nameof(GamBinaryTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = CreateDiscreteParameter("Label");

        [Parameter(nameof(GamBinaryTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = CreateDiscreteParameter("Features");

        [Parameter(nameof(GamBinaryTrainer.Options.NumberOfIterations))]
        public Parameter<int> NumberOfIterations = CreateInt32Parameter(100, 50000, true);

        [Parameter(nameof(GamBinaryTrainer.Options.MaximumBinCountPerFeature))]
        public Parameter<int> MaximumBinCountPerFeature = CreateInt32Parameter(10, 1000, true);

        [Parameter(nameof(GamBinaryTrainer.Options.LearningRate))]
        public Parameter<double> LearningRate = CreateDoubleParameter(1e-4, 1, true);
    }
}
