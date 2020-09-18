// <copyright file="OnlineGradientDescentTrainerSweepableOptions.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    public class OnlineGradientDescentTrainerSweepableOptions : SweepableOption<OnlineGradientDescentTrainer.Options>
    {
        public static OnlineGradientDescentTrainerSweepableOptions Default = new OnlineGradientDescentTrainerSweepableOptions();

        [Parameter(nameof(OnlineGradientDescentTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = CreateDiscreteParameter("Label");

        [Parameter(nameof(OnlineGradientDescentTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = CreateDiscreteParameter("Features");

        [Parameter(nameof(OnlineGradientDescentTrainer.Options.LossFunction))]
        public Parameter<ISupportSdcaClassificationLoss> LossFunction = CreateDiscreteParameter<ISupportSdcaClassificationLoss>(new HingeLoss(), new LogLoss(), new SmoothedHingeLoss());

        [Parameter(nameof(OnlineGradientDescentTrainer.Options.LearningRate))]
        public Parameter<float> LearningRate = CreateFloatParameter(1e-4f, 1e-1f, true);

        [Parameter(nameof(OnlineGradientDescentTrainer.Options.DecreaseLearningRate))]
        public Parameter<bool> DecreaseLearningRate = CreateDiscreteParameter(false, true);

        [Parameter(nameof(OnlineGradientDescentTrainer.Options.L2Regularization))]
        public Parameter<float> L2Regularization = CreateFloatParameter(1e-5f, 100f, true);

        [Parameter(nameof(OnlineGradientDescentTrainer.Options.NumberOfIterations))]
        public Parameter<int> NumberOfIterations = CreateInt32Parameter(1, 256, true);
    }
}
