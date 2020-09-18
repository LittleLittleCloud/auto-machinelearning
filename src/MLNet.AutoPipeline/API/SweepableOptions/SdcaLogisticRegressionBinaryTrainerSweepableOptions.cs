// <copyright file="SdcaLogisticRegressionBinaryTrainerSweepableOptions.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    public class SdcaLogisticRegressionBinaryTrainerSweepableOptions : SweepableOption<SdcaLogisticRegressionBinaryTrainer.Options>
    {
        public static SdcaLogisticRegressionBinaryTrainerSweepableOptions Default = new SdcaLogisticRegressionBinaryTrainerSweepableOptions();

        [Parameter(nameof(SdcaLogisticRegressionBinaryTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = CreateDiscreteParameter("Label");

        [Parameter(nameof(SdcaLogisticRegressionBinaryTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = CreateDiscreteParameter("Features");

        [Parameter(nameof(SdcaLogisticRegressionBinaryTrainer.Options.L1Regularization))]
        public Parameter<float> L1Regularization = CreateFloatParameter(1e-3f, 100f, true);

        [Parameter(nameof(SdcaLogisticRegressionBinaryTrainer.Options.L2Regularization))]
        public Parameter<float> L2Regularization = CreateFloatParameter(1e-5f, 100f, true);

        [Parameter(nameof(SdcaLogisticRegressionBinaryTrainer.Options.MaximumNumberOfIterations))]
        public Parameter<int> MaximumNumberOfIterations = CreateInt32Parameter(1, 256, true);
    }
}
