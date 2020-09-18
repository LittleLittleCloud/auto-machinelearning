// <copyright file="GamRegressionTrainerSweepableOptions.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers.FastTree;

namespace MLNet.AutoPipeline
{
    public class GamRegressionTrainerSweepableOptions : SweepableOption<GamRegressionTrainer.Options>
    {
        public static GamRegressionTrainerSweepableOptions Default = new GamRegressionTrainerSweepableOptions();

        [Parameter(nameof(GamRegressionTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = CreateDiscreteParameter("Label");

        [Parameter(nameof(GamRegressionTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = CreateDiscreteParameter("Features");

        [Parameter(nameof(GamRegressionTrainer.Options.NumberOfIterations))]
        public Parameter<int> NumberOfIterations = CreateInt32Parameter(100, 50000, true);

        [Parameter(nameof(GamRegressionTrainer.Options.MaximumBinCountPerFeature))]
        public Parameter<int> MaximumBinCountPerFeature = CreateInt32Parameter(10, 1000, true);

        [Parameter(nameof(GamRegressionTrainer.Options.LearningRate))]
        public Parameter<double> LearningRate = CreateDoubleParameter(1e-4, 1, true);
    }
}
