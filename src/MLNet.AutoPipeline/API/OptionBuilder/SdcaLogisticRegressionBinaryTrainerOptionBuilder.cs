// <copyright file="SdcaLogisticRegressionBinaryTrainerOptionBuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    public class SdcaLogisticRegressionBinaryTrainerOptionBuilder : OptionBuilder<SdcaLogisticRegressionBinaryTrainer.Options>
    {
        public static SdcaLogisticRegressionBinaryTrainerOptionBuilder Default = new SdcaLogisticRegressionBinaryTrainerOptionBuilder();

        [Parameter(nameof(SdcaLogisticRegressionBinaryTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = ParameterBuilder.CreateFromSingleValue("Label");

        [Parameter(nameof(SdcaLogisticRegressionBinaryTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = ParameterBuilder.CreateFromSingleValue("Features");

        [Parameter(nameof(SdcaLogisticRegressionBinaryTrainer.Options.ExampleWeightColumnName))]
        public Parameter<string> ExampleWeightColumnName = ParameterBuilder.CreateFromSingleValue<string>(default);

        [Parameter(nameof(SdcaLogisticRegressionBinaryTrainer.Options.L1Regularization))]
        public Parameter<float> L1Regularization = ParameterBuilder.CreateFloatParameter(1e-3f, 100f, true);

        [Parameter(nameof(SdcaLogisticRegressionBinaryTrainer.Options.L2Regularization))]
        public Parameter<float> L2Regularization = ParameterBuilder.CreateFloatParameter(1e-5f, 100f, true);

        [Parameter(nameof(SdcaLogisticRegressionBinaryTrainer.Options.MaximumNumberOfIterations))]
        public Parameter<int> MaximumNumberOfIterations = ParameterBuilder.CreateInt32Parameter(1, 256, true);
    }
}
