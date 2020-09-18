// <copyright file="SgdCalibratedBinaryTrainerSweepableOptions.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    public class SgdCalibratedBinaryTrainerSweepableOptions : SweepableOption<SgdCalibratedTrainer.Options>
    {
        public static SgdCalibratedBinaryTrainerSweepableOptions Default = new SgdCalibratedBinaryTrainerSweepableOptions();

        [Parameter(nameof(SgdCalibratedTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = CreateDiscreteParameter("Label");

        [Parameter(nameof(SgdCalibratedTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = CreateDiscreteParameter("Features");

        [Parameter(nameof(SgdCalibratedTrainer.Options.NumberOfIterations))]
        public Parameter<int> NumberOfIterations = CreateInt32Parameter(5, 200, true);

        [Parameter(nameof(SgdCalibratedTrainer.Options.LearningRate))]
        public Parameter<double> LearningRate = CreateDoubleParameter(1e-4, 1, true);

        [Parameter(nameof(SgdCalibratedTrainer.Options.L2Regularization))]
        public Parameter<float> L2Regularization = CreateFloatParameter(1e-7f, 1e-1f, true);
    }
}
