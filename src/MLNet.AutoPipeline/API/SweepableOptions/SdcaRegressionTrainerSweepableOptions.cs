// <copyright file="SdcaRegressionTrainerSweepableOptions.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    public class SdcaRegressionTrainerSweepableOptions : SweepableOption<SdcaRegressionTrainer.Options>
    {
        public static SdcaRegressionTrainerSweepableOptions Default = new SdcaRegressionTrainerSweepableOptions();

        [Parameter(nameof(SdcaRegressionTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = CreateDiscreteParameter("Label");

        [Parameter(nameof(SdcaRegressionTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = CreateDiscreteParameter("Features");

        [Parameter(nameof(SdcaRegressionTrainer.Options.LossFunction))]
        public Parameter<ISupportSdcaClassificationLoss> LossFunction = CreateDiscreteParameter<ISupportSdcaClassificationLoss>(new HingeLoss(), new LogLoss(), new SmoothedHingeLoss());

        [Parameter(nameof(SdcaRegressionTrainer.Options.L1Regularization))]
        public Parameter<float> L1Regularization = CreateFloatParameter(1e-3f, 100f, true);

        [Parameter(nameof(SdcaRegressionTrainer.Options.L2Regularization))]
        public Parameter<float> L2Regularization = CreateFloatParameter(1e-5f, 100f, true);

        [Parameter(nameof(SdcaRegressionTrainer.Options.MaximumNumberOfIterations))]
        public Parameter<int> MaximumNumberOfIterations = CreateInt32Parameter(1, 256, true);
    }
}
