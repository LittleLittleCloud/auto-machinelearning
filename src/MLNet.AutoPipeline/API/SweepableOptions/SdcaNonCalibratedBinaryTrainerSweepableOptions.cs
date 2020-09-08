// <copyright file="SdcaNonCalibratedBinaryTrainerSweepableOptions.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    public class SdcaNonCalibratedBinaryTrainerSweepableOptions : SweepableOption<SdcaNonCalibratedBinaryTrainer.Options>
    {
        public static SdcaNonCalibratedBinaryTrainerSweepableOptions Default = new SdcaNonCalibratedBinaryTrainerSweepableOptions();

        [Parameter(nameof(SdcaNonCalibratedBinaryTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = CreateDiscreteParameter("Label");

        [Parameter(nameof(SdcaNonCalibratedBinaryTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = CreateDiscreteParameter("Features");

        [Parameter(nameof(SdcaNonCalibratedBinaryTrainer.Options.LossFunction))]
        public Parameter<ISupportSdcaClassificationLoss> LossFunction = CreateDiscreteParameter<ISupportSdcaClassificationLoss>(new HingeLoss(), new LogLoss(), new SmoothedHingeLoss());

        [Parameter(nameof(SdcaNonCalibratedBinaryTrainer.Options.L1Regularization))]
        public Parameter<float> L1Regularization = CreateFloatParameter(1e-3f, 100f, true);

        [Parameter(nameof(SdcaNonCalibratedBinaryTrainer.Options.L2Regularization))]
        public Parameter<float> L2Regularization = CreateFloatParameter(1e-5f, 100f, true);

        [Parameter(nameof(SdcaNonCalibratedBinaryTrainer.Options.MaximumNumberOfIterations))]
        public Parameter<int> MaximumNumberOfIterations = CreateInt32Parameter(1, 256, true);
    }
}
