// <copyright file="SgdNonCalibratedBinaryTrainerSweepableOptions.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    public class SgdNonCalibratedBinaryTrainerSweepableOptions : SweepableOption<SgdNonCalibratedTrainer.Options>
    {
        public static SgdNonCalibratedBinaryTrainerSweepableOptions Default = new SgdNonCalibratedBinaryTrainerSweepableOptions();

        [Parameter(nameof(SgdNonCalibratedTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = CreateDiscreteParameter("Label");

        [Parameter(nameof(SgdNonCalibratedTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = CreateDiscreteParameter("Features");

        [Parameter(nameof(SgdNonCalibratedTrainer.Options.LossFunction))]
        public Parameter<IClassificationLoss> LossFunction = CreateDiscreteParameter<IClassificationLoss>(new HingeLoss(), new ExpLoss(), new LogLoss(), new SmoothedHingeLoss());

        [Parameter(nameof(SgdNonCalibratedTrainer.Options.NumberOfIterations))]
        public Parameter<int> NumberOfIterations = CreateInt32Parameter(5, 200, true);

        [Parameter(nameof(SgdNonCalibratedTrainer.Options.LearningRate))]
        public Parameter<double> LearningRate = CreateDoubleParameter(1e-4, 1, true);

        [Parameter(nameof(SgdNonCalibratedTrainer.Options.L2Regularization))]
        public Parameter<float> L2Regularization = CreateFloatParameter(1e-7f, 1e-1f, true);
    }
}
