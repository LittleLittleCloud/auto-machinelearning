// <copyright file="SgdNonCalibratedBinaryTrainerOptionBuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    public class SgdNonCalibratedBinaryTrainerOptionBuilder : SweepableOption<SgdNonCalibratedTrainer.Options>
    {
        public static SgdNonCalibratedBinaryTrainerOptionBuilder Default = new SgdNonCalibratedBinaryTrainerOptionBuilder();

        [Parameter(nameof(SgdNonCalibratedTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = ParameterBuilder.CreateFromSingleValue("Label");

        [Parameter(nameof(SgdNonCalibratedTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = ParameterBuilder.CreateFromSingleValue("Features");

        [Parameter(nameof(SgdNonCalibratedTrainer.Options.ExampleWeightColumnName))]
        public Parameter<string> ExampleWeightColumnName = ParameterBuilder.CreateFromSingleValue<string>(default);

        [Parameter(nameof(SgdNonCalibratedTrainer.Options.LossFunction))]
        public Parameter<IClassificationLoss> LossFunction = ParameterBuilder.CreateDiscreteParameter<IClassificationLoss>(new HingeLoss(), new ExpLoss(), new LogLoss(), new SmoothedHingeLoss());

        [Parameter(nameof(SgdNonCalibratedTrainer.Options.NumberOfIterations))]
        public Parameter<int> NumberOfIterations = ParameterBuilder.CreateInt32Parameter(5, 200, true);

        [Parameter(nameof(SgdNonCalibratedTrainer.Options.LearningRate))]
        public Parameter<double> LearningRate = ParameterBuilder.CreateDoubleParameter(1e-4, 1, true);

        [Parameter(nameof(SgdNonCalibratedTrainer.Options.L2Regularization))]
        public Parameter<float> L2Regularization = ParameterBuilder.CreateFloatParameter(1e-7f, 1e-1f, true);
    }
}
