// <copyright file="SdcaNonCalibratedBinaryTrainerOptionBuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    public class SdcaNonCalibratedBinaryTrainerOptionBuilder : SweepableOption<SdcaNonCalibratedBinaryTrainer.Options>
    {
        public static SdcaNonCalibratedBinaryTrainerOptionBuilder Default = new SdcaNonCalibratedBinaryTrainerOptionBuilder();

        [Parameter(nameof(SdcaNonCalibratedBinaryTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = ParameterBuilder.CreateFromSingleValue("Label");

        [Parameter(nameof(SdcaNonCalibratedBinaryTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = ParameterBuilder.CreateFromSingleValue("Features");

        [Parameter(nameof(SdcaNonCalibratedBinaryTrainer.Options.ExampleWeightColumnName))]
        public Parameter<string> ExampleWeightColumnName = ParameterBuilder.CreateFromSingleValue<string>(default);

        [Parameter(nameof(SdcaNonCalibratedBinaryTrainer.Options.LossFunction))]
        public Parameter<ISupportSdcaClassificationLoss> LossFunction = ParameterBuilder.CreateDiscreteParameter<ISupportSdcaClassificationLoss>(new HingeLoss(), new LogLoss(), new SmoothedHingeLoss());

        [Parameter(nameof(SdcaNonCalibratedBinaryTrainer.Options.L1Regularization))]
        public Parameter<float> L1Regularization = ParameterBuilder.CreateFloatParameter(1e-3f, 100f, true);

        [Parameter(nameof(SdcaNonCalibratedBinaryTrainer.Options.L2Regularization))]
        public Parameter<float> L2Regularization = ParameterBuilder.CreateFloatParameter(1e-5f, 100f, true);

        [Parameter(nameof(SdcaNonCalibratedBinaryTrainer.Options.MaximumNumberOfIterations))]
        public Parameter<int> MaximumNumberOfIterations = ParameterBuilder.CreateInt32Parameter(1, 256, true);
    }
}
