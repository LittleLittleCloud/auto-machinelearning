using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
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
            LinearSvm(
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

        public static SweepableNode<LdSvmTrainer, LdSvmTrainer.Options>
            LdSvm(
                this SweepableBinaryClassificationTrainers trainer,
                string labelColumnName = "Label",
                string featureColumnName = "Features",
                OptionBuilder<LdSvmTrainer.Options> optionBuilder = null,
                LdSvmTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = LdSvmOptionBuilder.Default;
            }

            optionBuilder.SetDefaultOption(defaultOption);

            return context.AutoML().SweepableTrainer(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.BinaryClassification.Trainers.LdSvm(option);
                },
                optionBuilder,
                new string[] { featureColumnName },
                PredictedLabel,
                nameof(LdSvmTrainer));
        }

        public static SweepableNode<FastForestBinaryTrainer, FastForestBinaryTrainer.Options>
            FastForest(
                this SweepableBinaryClassificationTrainers trainer,
                string labelColumnName = "Label",
                string featureColumnName = "Features",
                OptionBuilder<FastForestBinaryTrainer.Options> optionBuilder = null,
                FastForestBinaryTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = FastForestBinaryTrainerOptionBuilder.Default;
            }

            optionBuilder.SetDefaultOption(defaultOption);
            return context.AutoML().SweepableTrainer(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.BinaryClassification.Trainers.FastForest(option);
                },
                optionBuilder,
                new string[] { featureColumnName },
                PredictedLabel,
                nameof(FastForestBinaryTrainer));
        }

        public static SweepableNode<FastTreeBinaryTrainer, FastTreeBinaryTrainer.Options>
            FastTree(
                this SweepableBinaryClassificationTrainers trainer,
                string labelColumnName = "Label",
                string featureColumnName = "Features",
                OptionBuilder<FastTreeBinaryTrainer.Options> optionBuilder = null,
                FastTreeBinaryTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = FastTreeBinaryTrainerOptionBuilder.Default;
            }

            optionBuilder.SetDefaultOption(defaultOption);
            return context.AutoML().SweepableTrainer(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.BinaryClassification.Trainers.FastTree(option);
                },
                optionBuilder,
                new string[] { featureColumnName },
                PredictedLabel,
                nameof(FastTreeBinaryTrainer));
        }

        public static SweepableNode<LightGbmBinaryTrainer, LightGbmBinaryTrainer.Options>
            LightGbm(
                this SweepableBinaryClassificationTrainers trainer,
                string labelColumnName = "Label",
                string featureColumnName = "Features",
                OptionBuilder<LightGbmBinaryTrainer.Options> optionBuilder = null,
                LightGbmBinaryTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = LightGbmBinaryClassOptionBuilder.Default;
            }

            optionBuilder.SetDefaultOption(defaultOption);
            return context.AutoML().SweepableTrainer(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.BinaryClassification.Trainers.LightGbm(option);
                },
                optionBuilder,
                new string[] { featureColumnName },
                PredictedLabel,
                nameof(FastTreeBinaryTrainer));
        }
    }
}
