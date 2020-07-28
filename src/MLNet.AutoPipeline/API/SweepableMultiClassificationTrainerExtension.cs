// <copyright file="SweepableMultiClassificationTrainerExtension.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.LightGbm;
using MLNet.AutoPipeline.API.OptionBuilder;
using System;
using System.Collections.Generic;
using System.Reflection.Emit;
using System.Text;

namespace MLNet.AutoPipeline
{
    /// <summary>
    /// class contains extension method for <see cref="SweepableMultiClassificationTrainers"/>.
    /// </summary>
    public static class SweepableMultiClassificationTrainerExtension
    {
        /// <summary>
        /// Create an <see cref="UnsweepableNode{SweepableBinaryClassificationTrainers}"/> where TTrainer is <see cref="NaiveBayesMulticlassTrainer"/> that can be used in <see cref="ISweepablePipeline"/>.
        /// </summary>
        /// <param name="trainer">The <see cref="SweepableMultiClassificationTrainerExtension"/>.</param>
        /// <param name="labelColumnName">label column name. Default is Label.</param>
        /// <param name="featureColumnName">feature column name. Default is Features.</param>
        /// <returns><see cref="UnsweepableNode{TTrainer}"/>.</returns>
        public static UnsweepableNode<NaiveBayesMulticlassTrainer>
            NaiveBayes(this SweepableMultiClassificationTrainers trainer, string labelColumnName = "Label", string featureColumnName = "Features")
        {
            var context = trainer.Context;
            var instance = context.MulticlassClassification.Trainers.NaiveBayes(labelColumnName, featureColumnName);
            return Util.CreateUnSweepableNode(
                        instance,
                        estimatorName: nameof(NaiveBayesMulticlassTrainer),
                        inputs: new string[] { labelColumnName },
                        outputs: new string[] { featureColumnName });
        }

        /// <summary>
        /// Create a <see cref="SweepableNode{TNewTrain, TOption}"/> where TNewTrain is <see cref="SdcaMaximumEntropyMulticlassTrainer"/> and TOption is <see cref="SdcaMaximumEntropyMulticlassTrainer.Options"/> that can be used in <see cref="ISweepablePipeline"/>.
        /// </summary>
        /// <param name="trainer">The <see cref="SweepableMultiClassificationTrainerExtension"/>.</param>
        /// <param name="labelColumnName">label column name. Default is Label.</param>
        /// <param name="featureColumnName">feature column name, Default is Features.</param>
        /// <param name="optionBuilder">option builder. if null, a default instance of <see cref="SdcaMaximumEntropyOptionBuilder"/> will be used.</param>
        /// <param name="defaultOption">predefined option. if null, default option will be used.</param>
        /// <returns><see cref="SweepableNode{TNewTrain, TOption}"/>.</returns>
        public static SweepableNode<SdcaMaximumEntropyMulticlassTrainer, SdcaMaximumEntropyMulticlassTrainer.Options>
            SdcaMaximumEntropy(this SweepableMultiClassificationTrainers trainer, string labelColumnName = "Label", string featureColumnName = "Features", OptionBuilder<SdcaMaximumEntropyMulticlassTrainer.Options> optionBuilder = null, SdcaMaximumEntropyMulticlassTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = SdcaMaximumEntropyOptionBuilder.Default;
            }

            return Util.CreateSweepableNode(
                                (option) =>
                                {
                                    if (defaultOption != null)
                                    {
                                        Util.CopyFieldsTo(defaultOption, option);
                                    }

                                    option.LabelColumnName = labelColumnName;
                                    option.FeatureColumnName = featureColumnName;
                                    return context.MulticlassClassification.Trainers.SdcaMaximumEntropy(option);
                                },
                                optionBuilder,
                                estimatorName: nameof(SdcaMaximumEntropyMulticlassTrainer),
                                inputs: new string[] { labelColumnName },
                                outputs: new string[] { featureColumnName });
        }

        /// <summary>
        /// Create a <see cref="SweepableNode{TNewTrain, TOption}"/> where TNewTrain is <see cref="SdcaNonCalibratedMulticlassTrainer"/> and TOption is <see cref="SdcaNonCalibratedMulticlassTrainer.Options"/> that can be used in <see cref="ISweepablePipeline"/>.
        /// </summary>
        /// <param name="trainer">The <see cref="SweepableMultiClassificationTrainerExtension"/>.</param>
        /// <param name="labelColumnName">label column name. Default is Label.</param>
        /// <param name="featureColumnName">feature column name, Default is Features.</param>
        /// <param name="optionBuilder">option builder. if null, a default instance of <see cref="SdcaNonCalibratedOptionBuilder"/> will be used.</param>
        /// <param name="defaultOption">predefined option. if null, default option will be used.</param>
        /// <returns><see cref="SweepableNode{TNewTrain, TOption}"/>.</returns>
        public static SweepableNode<SdcaNonCalibratedMulticlassTrainer, SdcaNonCalibratedMulticlassTrainer.Options>
                SdcaNonCalibreated(this SweepableMultiClassificationTrainers trainer, string labelColumnName = "Label", string featureColumnName = "Features", OptionBuilder<SdcaNonCalibratedMulticlassTrainer.Options> optionBuilder = null, SdcaNonCalibratedMulticlassTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = SdcaNonCalibratedOptionBuilder.Default;
            }

            return Util.CreateSweepableNode(
                                (option) =>
                                {
                                    if (defaultOption != null)
                                    {
                                        Util.CopyFieldsTo(defaultOption, option);
                                    }

                                    option.LabelColumnName = labelColumnName;
                                    option.FeatureColumnName = featureColumnName;
                                    return context.MulticlassClassification.Trainers.SdcaNonCalibrated(option);
                                },
                                optionBuilder,
                                estimatorName: nameof(SdcaNonCalibratedMulticlassTrainer),
                                inputs: new string[] { labelColumnName },
                                outputs: new string[] { featureColumnName });
        }

        /// <summary>
        /// Create a <see cref="SweepableNode{TNewTrain, TOption}"/> where TNewTrain is <see cref="LbfgsMaximumEntropyMulticlassTrainer"/> and TOption is <see cref="LbfgsMaximumEntropyMulticlassTrainer.Options"/> that can be used in <see cref="ISweepablePipeline"/>.
        /// </summary>
        /// <param name="trainer">The <see cref="SweepableMultiClassificationTrainerExtension"/>.</param>
        /// <param name="labelColumnName">label column name. Default is Label.</param>
        /// <param name="featureColumnName">feature column name, Default is Features.</param>
        /// <param name="optionBuilder">option builder. if null, a default instance of <see cref="LbfgsMaximumEntropyOptionBuilder"/> will be used.</param>
        /// <param name="defaultOption">predefined option. if null, default option will be used.</param>
        /// <returns><see cref="SweepableNode{TNewTrain, TOption}"/>.</returns>
        public static SweepableNode<LbfgsMaximumEntropyMulticlassTrainer, LbfgsMaximumEntropyMulticlassTrainer.Options>
                LbfgsMaximumEntropy(this SweepableMultiClassificationTrainers trainer, string labelColumnName = "Label", string featureColumnName = "Features", OptionBuilder<LbfgsMaximumEntropyMulticlassTrainer.Options> optionBuilder = null, LbfgsMaximumEntropyMulticlassTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = LbfgsMaximumEntropyOptionBuilder.Default;
            }

            return Util.CreateSweepableNode(
                                (option) =>
                                {
                                    if (defaultOption != null)
                                    {
                                        Util.CopyFieldsTo(defaultOption, option);
                                    }

                                    option.LabelColumnName = labelColumnName;
                                    option.FeatureColumnName = featureColumnName;
                                    return context.MulticlassClassification.Trainers.LbfgsMaximumEntropy(option);
                                },
                                optionBuilder,
                                estimatorName: nameof(LbfgsMaximumEntropyMulticlassTrainer),
                                inputs: new string[] { labelColumnName },
                                outputs: new string[] { featureColumnName });
        }

        /// <summary>
        /// Create a <see cref="SweepableNode{TNewTrain, TOption}"/> where TNewTrain is <see cref="LightGbmMulticlassTrainer"/> and TOption is <see cref="LightGbmMulticlassTrainer.Options"/> that can be used in <see cref="ISweepablePipeline"/>.
        /// </summary>
        /// <param name="trainer">The <see cref="SweepableMultiClassificationTrainerExtension"/>.</param>
        /// <param name="labelColumnName">label column name. Default is Label.</param>
        /// <param name="featureColumnName">feature column name, Default is Features.</param>
        /// <param name="optionBuilder">option builder. if null, a default instance of <see cref="LightGbmOptionBuilder"/> will be used.</param>
        /// <param name="defaultOption">predefined option. if null, default option will be used.</param>
        /// <returns><see cref="SweepableNode{TNewTrain, TOption}"/>.</returns>
        public static SweepableNode<LightGbmMulticlassTrainer, LightGbmMulticlassTrainer.Options>
            LightGbm(this SweepableMultiClassificationTrainers trainer, string labelColumnName = "Label", string featureColumnName = "Features", OptionBuilder<LightGbmMulticlassTrainer.Options> optionBuilder = null, LightGbmMulticlassTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = LightGbmOptionBuilder.Default;
            }

            return Util.CreateSweepableNode(
                                (option) =>
                                {
                                    if (defaultOption != null)
                                    {
                                        Util.CopyFieldsTo(defaultOption, option);
                                    }

                                    option.LabelColumnName = labelColumnName;
                                    option.FeatureColumnName = featureColumnName;
                                    return context.MulticlassClassification.Trainers.LightGbm(option);
                                },
                                optionBuilder,
                                estimatorName: nameof(LightGbmMulticlassTrainer),
                                inputs: new string[] { labelColumnName },
                                outputs: new string[] { featureColumnName });
        }
    }
}
