// <copyright file="SgdCalibratedBinaryTrainerOptionBuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    public class SgdCalibratedBinaryTrainerOptionBuilder : SweepableOption<SgdCalibratedTrainer.Options>
    {
        public static SgdCalibratedBinaryTrainerOptionBuilder Default = new SgdCalibratedBinaryTrainerOptionBuilder();

        [Parameter(nameof(SgdCalibratedTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = ParameterBuilder.CreateFromSingleValue("Label");

        [Parameter(nameof(SgdCalibratedTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = ParameterBuilder.CreateFromSingleValue("Features");

        [Parameter(nameof(SgdCalibratedTrainer.Options.ExampleWeightColumnName))]
        public Parameter<string> ExampleWeightColumnName = ParameterBuilder.CreateFromSingleValue<string>(default);

        [Parameter(nameof(SgdCalibratedTrainer.Options.NumberOfIterations))]
        public Parameter<int> NumberOfIterations = ParameterBuilder.CreateInt32Parameter(5, 200, true);

        [Parameter(nameof(SgdCalibratedTrainer.Options.LearningRate))]
        public Parameter<double> LearningRate = ParameterBuilder.CreateDoubleParameter(1e-4, 1, true);

        [Parameter(nameof(SgdCalibratedTrainer.Options.L2Regularization))]
        public Parameter<float> L2Regularization = ParameterBuilder.CreateFloatParameter(1e-7f, 1e-1f, true);
    }
}
