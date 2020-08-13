// <copyright file="AveragedPerceptronBinaryTrainerOptionBuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    public class AveragedPerceptronBinaryTrainerOptionBuilder : OptionBuilder<AveragedPerceptronTrainer.Options>
    {
        public static AveragedPerceptronBinaryTrainerOptionBuilder Default = new AveragedPerceptronBinaryTrainerOptionBuilder();

        [Parameter(nameof(AveragedPerceptronTrainer.Options.LabelColumnName))]
        public Parameter<string> LabelColumnName = ParameterBuilder.CreateFromSingleValue("Label");

        [Parameter(nameof(AveragedPerceptronTrainer.Options.FeatureColumnName))]
        public Parameter<string> FeatureColumnName = ParameterBuilder.CreateFromSingleValue("Features");

        [Parameter(nameof(AveragedPerceptronTrainer.Options.LossFunction))]
        public Parameter<IClassificationLoss> LossFunction = ParameterBuilder.CreateDiscreteParameter<IClassificationLoss>(new HingeLoss(), new ExpLoss(), new LogLoss(), new SmoothedHingeLoss());

        [Parameter(nameof(AveragedPerceptronTrainer.Options.L2Regularization))]
        public Parameter<float> L2Regularization = ParameterBuilder.CreateFloatParameter(1e-5f, 100f, true);

        [Parameter(nameof(AveragedPerceptronTrainer.Options.LearningRate))]
        public Parameter<float> LearningRate = ParameterBuilder.CreateFloatParameter(1e-5f, 100f, true);

        [Parameter(nameof(AveragedPerceptronTrainer.Options.NumberOfIterations))]
        public Parameter<int> NumberOfIterations = ParameterBuilder.CreateInt32Parameter(1, 256, true);

        [Parameter(nameof(AveragedPerceptronTrainer.Options.DecreaseLearningRate))]
        public Parameter<bool> DecreaseLearningRate = ParameterBuilder.CreateFromSingleValue(false);
    }
}
