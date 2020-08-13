// <copyright file="GamBinaryTrainerOptionBuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers.FastTree;

namespace MLNet.AutoPipeline
{
    public class GamBinaryTrainerOptionBuilder: OptionBuilder<GamBinaryTrainer.Options>
    {
        public static GamBinaryTrainerOptionBuilder Default = new GamBinaryTrainerOptionBuilder();

        [Parameter(nameof(GamBinaryTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = ParameterBuilder.CreateFromSingleValue("Label");

        [Parameter(nameof(GamBinaryTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = ParameterBuilder.CreateFromSingleValue("Features");

        [Parameter(nameof(GamBinaryTrainer.Options.ExampleWeightColumnName))]
        public Parameter<string> ExampleWeightColumnName = ParameterBuilder.CreateFromSingleValue<string>(default);

        [Parameter(nameof(GamBinaryTrainer.Options.NumberOfIterations))]
        public Parameter<int> NumberOfIterations = ParameterBuilder.CreateInt32Parameter(100, 50000, true);

        [Parameter(nameof(GamBinaryTrainer.Options.MaximumBinCountPerFeature))]
        public Parameter<int> MaximumBinCountPerFeature = ParameterBuilder.CreateInt32Parameter(10, 1000, true);

        [Parameter(nameof(GamBinaryTrainer.Options.LearningRate))]
        public Parameter<double> LearningRate = ParameterBuilder.CreateDoubleParameter(1e-4, 1, true);
    }
}
