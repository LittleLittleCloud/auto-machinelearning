// <copyright file="SweepableMultiClassificationTrainerExtension.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.LightGbm;
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
        /// Create an <see cref="UnsweepableNode{SweepableBinaryClassificationTrainers}"/> where TTrainer is <see cref="NaiveBayesMulticlassTrainer"/> that can be used in <see cref="SweepablePipeline"/>.
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
            return context.AutoML().UnsweepableTrainer(
                        instance,
                        estimatorName: nameof(NaiveBayesMulticlassTrainer),
                        inputs: new string[] { labelColumnName },
                        outputs: new string[] { featureColumnName });
        }

        /// <summary>
        /// Create a <see cref="SweepableNode{TNewTrain, TOption}"/> where TNewTrain is <see cref="SdcaMaximumEntropyMulticlassTrainer"/> and TOption is <see cref="SdcaMaximumEntropyMulticlassTrainer.Options"/> that can be used in <see cref="SweepablePipeline"/>.
        /// </summary>
        /// <param name="trainer">The <see cref="SweepableMultiClassificationTrainerExtension"/>.</param>
        /// <param name="labelColumnName">label column name. Default is Label.</param>
        /// <param name="featureColumnName">feature column name, Default is Features.</param>
        /// <param name="optionBuilder">option builder. if null, a default instance of <see cref="SdcaMaximumEntropyMulticlassTrainerSweepableOptions"/> will be used.</param>
        /// <param name="defaultOption">predefined option. if null, default option will be used.</param>
        /// <returns><see cref="SweepableNode{TNewTrain, TOption}"/>.</returns>
        public static SweepableNode<SdcaMaximumEntropyMulticlassTrainer, SdcaMaximumEntropyMulticlassTrainer.Options>
            SdcaMaximumEntropy(this SweepableMultiClassificationTrainers trainer, string labelColumnName = "Label", string featureColumnName = "Features", SweepableOption<SdcaMaximumEntropyMulticlassTrainer.Options> optionBuilder = null, SdcaMaximumEntropyMulticlassTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = SdcaMaximumEntropyMulticlassTrainerSweepableOptions.Default;
            }

            optionBuilder.SetDefaultOption(defaultOption);

            return context.AutoML().SweepableTrainer(
                                (context, option) =>
                                {
                                    option.LabelColumnName = labelColumnName;
                                    option.FeatureColumnName = featureColumnName;
                                    return context.MulticlassClassification.Trainers.SdcaMaximumEntropy(option);
                                },
                                optionBuilder,
                                new string[] { labelColumnName },
                                new string[] { featureColumnName },
                                nameof(SdcaMaximumEntropyMulticlassTrainer));
        }

        /// <summary>
        /// Create a <see cref="SweepableNode{TNewTrain, TOption}"/> where TNewTrain is <see cref="SdcaNonCalibratedMulticlassTrainer"/> and TOption is <see cref="SdcaNonCalibratedMulticlassTrainer.Options"/> that can be used in <see cref="SweepablePipeline"/>.
        /// </summary>
        /// <param name="trainer">The <see cref="SweepableMultiClassificationTrainerExtension"/>.</param>
        /// <param name="labelColumnName">label column name. Default is Label.</param>
        /// <param name="featureColumnName">feature column name, Default is Features.</param>
        /// <param name="optionBuilder">option builder. if null, a default instance of <see cref="SdcaNonCalibratedMulticlassTrainerSweepableOptions"/> will be used.</param>
        /// <param name="defaultOption">predefined option. if null, default option will be used.</param>
        /// <returns><see cref="SweepableNode{TNewTrain, TOption}"/>.</returns>
        public static SweepableNode<SdcaNonCalibratedMulticlassTrainer, SdcaNonCalibratedMulticlassTrainer.Options>
                SdcaNonCalibreated(this SweepableMultiClassificationTrainers trainer, string labelColumnName = "Label", string featureColumnName = "Features", SweepableOption<SdcaNonCalibratedMulticlassTrainer.Options> optionBuilder = null, SdcaNonCalibratedMulticlassTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = SdcaNonCalibratedMulticlassTrainerSweepableOptions.Default;
            }

            optionBuilder.SetDefaultOption(defaultOption);

            return context.AutoML().SweepableTrainer(
                                (context, option) =>
                                {
                                    option.LabelColumnName = labelColumnName;
                                    option.FeatureColumnName = featureColumnName;
                                    return context.MulticlassClassification.Trainers.SdcaNonCalibrated(option);
                                },
                                optionBuilder,
                                trainerName: nameof(SdcaNonCalibratedMulticlassTrainer),
                                inputs: new string[] { labelColumnName },
                                outputs: new string[] { featureColumnName });
        }

        /// <summary>
        /// Create a <see cref="SweepableNode{TNewTrain, TOption}"/> where TNewTrain is <see cref="LbfgsMaximumEntropyMulticlassTrainer"/> and TOption is <see cref="LbfgsMaximumEntropyMulticlassTrainer.Options"/> that can be used in <see cref="SweepablePipeline"/>.
        /// </summary>
        /// <param name="trainer">The <see cref="SweepableMultiClassificationTrainerExtension"/>.</param>
        /// <param name="labelColumnName">label column name. Default is Label.</param>
        /// <param name="featureColumnName">feature column name, Default is Features.</param>
        /// <param name="optionBuilder">option builder. if null, a default instance of <see cref="LbfgsMaximumEntropyMulticlassTrainerSweepableOptions"/> will be used.</param>
        /// <param name="defaultOption">predefined option. if null, default option will be used.</param>
        /// <returns><see cref="SweepableNode{TNewTrain, TOption}"/>.</returns>
        public static SweepableNode<LbfgsMaximumEntropyMulticlassTrainer, LbfgsMaximumEntropyMulticlassTrainer.Options>
                LbfgsMaximumEntropy(this SweepableMultiClassificationTrainers trainer, string labelColumnName = "Label", string featureColumnName = "Features", SweepableOption<LbfgsMaximumEntropyMulticlassTrainer.Options> optionBuilder = null, LbfgsMaximumEntropyMulticlassTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = LbfgsMaximumEntropyMulticlassTrainerSweepableOptions.Default;
            }

            optionBuilder.SetDefaultOption(defaultOption);

            return context.AutoML().SweepableTrainer(
                                (context, option) =>
                                {
                                    option.LabelColumnName = labelColumnName;
                                    option.FeatureColumnName = featureColumnName;
                                    return context.MulticlassClassification.Trainers.LbfgsMaximumEntropy(option);
                                },
                                optionBuilder,
                                trainerName: nameof(LbfgsMaximumEntropyMulticlassTrainer),
                                inputs: new string[] { labelColumnName },
                                outputs: new string[] { featureColumnName });
        }

        /// <summary>
        /// Create a <see cref="SweepableNode{TNewTrain, TOption}"/> where TNewTrain is <see cref="LightGbmMulticlassTrainer"/> and TOption is <see cref="LightGbmMulticlassTrainer.Options"/> that can be used in <see cref="SweepablePipeline"/>.
        /// </summary>
        /// <param name="trainer">The <see cref="SweepableMultiClassificationTrainerExtension"/>.</param>
        /// <param name="labelColumnName">label column name. Default is Label.</param>
        /// <param name="featureColumnName">feature column name, Default is Features.</param>
        /// <param name="optionBuilder">option builder. if null, a default instance of <see cref="LightGbmMulticlassTrainerSweepableOptions"/> will be used.</param>
        /// <param name="defaultOption">predefined option. if null, default option will be used.</param>
        /// <returns><see cref="SweepableNode{TNewTrain, TOption}"/>.</returns>
        public static SweepableNode<LightGbmMulticlassTrainer, LightGbmMulticlassTrainer.Options>
            LightGbm(this SweepableMultiClassificationTrainers trainer, string labelColumnName = "Label", string featureColumnName = "Features", SweepableOption<LightGbmMulticlassTrainer.Options> optionBuilder = null, LightGbmMulticlassTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = LightGbmMulticlassTrainerSweepableOptions.Default;
            }

            optionBuilder.SetDefaultOption(defaultOption);

            return context.AutoML().SweepableTrainer(
                                (context, option) =>
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
                                trainerName: nameof(LightGbmMulticlassTrainer),
                                inputs: new string[] { labelColumnName },
                                outputs: new string[] { featureColumnName });
        }

        /// <summary>
        /// Create a <see cref="SweepableNode{TNewTrain, TOption}"/> where TNewTrain is <see cref="OneVersusAllTrainer"/> that can be used in <see cref="SweepablePipeline"/>. This function is used to convert binary classification trainer into multi-classification trainer. 
        /// </summary>
        /// <typeparam name="TModel">Type of binary classification trainer.</typeparam>
        /// <typeparam name="TOption">type of binary classification trainer option.</typeparam>
        /// <param name="trainer">The <see cref="SweepableMultiClassificationTrainerExtension"/>.</param>
        /// <param name="node">An instance of a binary <see cref="SweepableNode{TTrain, TOption}"/> used as the base trainer.</param>
        /// <param name="labelColumnName">label column name.</param>
        /// <param name="imputeMissingLabelsAsNegative">Whether to treat missing labels as having negative labels, instead of keeping them missing.</param>
        /// <param name="calibrator">The calibrator. If a calibrator is not explicitly provided, it will default to Microsoft.ML.Calibrators.PlattCalibratorTrainer.</param>
        /// <param name="maximumCalibrationExampleCount">Number of instances to train the calibrator.</param>
        /// <param name="useProbabilities">se probabilities (vs. raw outputs) to identify top-score category.</param>
        /// <returns><see cref="SweepableNode{TNewTrain, TOption}"/>.</returns>
        public static SweepableNode<OneVersusAllTrainer, TOption>
            OneVersusAll<TModel, TOption>(this SweepableMultiClassificationTrainers trainer, ISweepableNode<ITrainerEstimator<BinaryPredictionTransformer<TModel>, TModel>, TOption> node, string labelColumnName = "Label", bool imputeMissingLabelsAsNegative = false, Microsoft.ML.IEstimator<Microsoft.ML.ISingleFeaturePredictionTransformer<Microsoft.ML.Calibrators.ICalibrator>> calibrator = default, int maximumCalibrationExampleCount = 1000000000, bool useProbabilities = true)
                where TModel : class
                where TOption : class
        {
            var context = trainer.Context;

            return context.AutoML().SweepableTrainer(
                                (context, option) =>
                                {
                                    var estimator = node.EstimatorFactory(option);
                                    return context.MulticlassClassification.Trainers.OneVersusAll(estimator, labelColumnName, imputeMissingLabelsAsNegative, calibrator, maximumCalibrationExampleCount, useProbabilities);
                                },
                                node.OptionBuilder,
                                inputs: node.InputColumns,
                                outputs: node.OutputColumns,
                                trainerName: node.EstimatorName + "Ova");
        }
    }
}
