﻿using Microsoft.ML;
using Microsoft.ML.Trainers;
using MLNet.AutoPipeline.API.OptionBuilder;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    public static class SweepableBinaryClassificationTrainerExtension
    {
        private static string PredictedLabel = "PredictedLabel";

        public static SweepableNode<LinearSvmTrainer, LinearSvmTrainer.Options>
            LinearSVM(
                this SweepableBinaryClassificationTrainers trainer,
                string labelColumnName = "Label",
                string featureColumnName = "Features",
                OptionBuilder<LinearSvmTrainer.Options> optionBuilder = null,
                LinearSvmTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = LinearSvmOptionBuilder.Default;
            }

            optionBuilder.SetDefaultOption(defaultOption);

            return context.AutoML().SweepableTrainer(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;
                    return context.BinaryClassification.Trainers.LinearSvm(option);
                },
                optionBuilder,
                new string[] { featureColumnName },
                PredictedLabel,
                nameof(LinearSvmTrainer));
        }
    }
}