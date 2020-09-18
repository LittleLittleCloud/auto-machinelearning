// <copyright file="AveragedPerceptronBinaryTrainerSweepableOptions.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    public class AveragedPerceptronBinaryTrainerSweepableOptions : SweepableOption<AveragedPerceptronTrainer.Options>
    {
        public static AveragedPerceptronBinaryTrainerSweepableOptions Default = new AveragedPerceptronBinaryTrainerSweepableOptions();

        [Parameter(nameof(AveragedPerceptronTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = CreateDiscreteParameter("Label");

        [Parameter(nameof(AveragedPerceptronTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = CreateDiscreteParameter("Features");

        [Parameter(nameof(AveragedPerceptronTrainer.Options.LossFunction))]
        public Parameter<IClassificationLoss> LossFunction = CreateDiscreteParameter<IClassificationLoss>(new HingeLoss(), new ExpLoss(), new LogLoss(), new SmoothedHingeLoss());

        [Parameter(nameof(AveragedPerceptronTrainer.Options.L2Regularization))]
        public Parameter<float> L2Regularization = CreateFloatParameter(1e-5f, 100f, true);

        [Parameter(nameof(AveragedPerceptronTrainer.Options.LearningRate))]
        public Parameter<float> LearningRate = CreateFloatParameter(1e-5f, 100f, true);

        [Parameter(nameof(AveragedPerceptronTrainer.Options.NumberOfIterations))]
        public Parameter<int> NumberOfIterations = CreateInt32Parameter(1, 256, true);

        [Parameter(nameof(AveragedPerceptronTrainer.Options.DecreaseLearningRate))]
        public Parameter<bool> DecreaseLearningRate = CreateDiscreteParameter(false);
    }
}
