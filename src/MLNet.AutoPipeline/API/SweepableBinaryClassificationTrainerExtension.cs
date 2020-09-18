// <copyright file="SweepableBinaryClassificationTrainerExtension.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    public static class SweepableBinaryClassificationTrainerExtension
    {
        private const string PredictedLabel = "PredictedLabel";

        public static SweepableEstimator<LinearSvmTrainer, LinearSvmTrainer.Options>
            LinearSvm(
                this SweepableBinaryClassificationTrainers trainer,
                string labelColumnName = "Label",
                string featureColumnName = "Features",
                SweepableOption<LinearSvmTrainer.Options> optionBuilder = null,
                LinearSvmTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = LinearSvmBinaryTrainerSweepableOptions.Default;
            }

            optionBuilder.SetDefaultOption(defaultOption);
            return context.AutoML().CreateSweepableEstimator(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;
                    return context.BinaryClassification.Trainers.LinearSvm(option);
                },
                optionBuilder,
                new string[] { labelColumnName, featureColumnName },
                new string[] { PredictedLabel },
                nameof(LinearSvmTrainer));
        }

        public static SweepableEstimator<LdSvmTrainer, LdSvmTrainer.Options>
            LdSvm(
                this SweepableBinaryClassificationTrainers trainer,
                string labelColumnName = "Label",
                string featureColumnName = "Features",
                SweepableOption<LdSvmTrainer.Options> optionBuilder = null,
                LdSvmTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = LdSvmBinaryTrainerSweepableOptions.Default;
            }

            optionBuilder.SetDefaultOption(defaultOption);

            return context.AutoML().CreateSweepableEstimator(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.BinaryClassification.Trainers.LdSvm(option);
                },
                optionBuilder,
                new string[] { labelColumnName, featureColumnName },
                new string[] { PredictedLabel },
                nameof(LdSvmTrainer));
        }

        public static SweepableEstimator<FastForestBinaryTrainer, FastForestBinaryTrainer.Options>
            FastForest(
                this SweepableBinaryClassificationTrainers trainer,
                string labelColumnName = "Label",
                string featureColumnName = "Features",
                SweepableOption<FastForestBinaryTrainer.Options> optionBuilder = null,
                FastForestBinaryTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = FastForestBinaryTrainerSweepableOptions.Default;
            }

            optionBuilder.SetDefaultOption(defaultOption);
            return context.AutoML().CreateSweepableEstimator(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.BinaryClassification.Trainers.FastForest(option);
                },
                optionBuilder,
                new string[] { labelColumnName, featureColumnName },
                new string[] { PredictedLabel },
                nameof(FastForestBinaryTrainer));
        }

        public static SweepableEstimator<FastTreeBinaryTrainer, FastTreeBinaryTrainer.Options>
            FastTree(
                this SweepableBinaryClassificationTrainers trainer,
                string labelColumnName = "Label",
                string featureColumnName = "Features",
                SweepableOption<FastTreeBinaryTrainer.Options> optionBuilder = null,
                FastTreeBinaryTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = FastTreeBinaryTrainerSweepableOptions.Default;
            }

            optionBuilder.SetDefaultOption(defaultOption);
            return context.AutoML().CreateSweepableEstimator(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.BinaryClassification.Trainers.FastTree(option);
                },
                optionBuilder,
                new string[] { labelColumnName, featureColumnName },
                new string[] { PredictedLabel },
                nameof(FastTreeBinaryTrainer));
        }

        public static SweepableEstimator<LightGbmBinaryTrainer, LightGbmBinaryTrainer.Options>
            LightGbm(
                this SweepableBinaryClassificationTrainers trainer,
                string labelColumnName = "Label",
                string featureColumnName = "Features",
                SweepableOption<LightGbmBinaryTrainer.Options> optionBuilder = null,
                LightGbmBinaryTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = LightGbmBinaryTrainerSweepableOptions.Default;
            }

            optionBuilder.SetDefaultOption(defaultOption);
            return context.AutoML().CreateSweepableEstimator(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.BinaryClassification.Trainers.LightGbm(option);
                },
                optionBuilder,
                new string[] { labelColumnName, featureColumnName },
                new string[] { PredictedLabel },
                nameof(LightGbmBinaryTrainer));
        }

        public static SweepableEstimator<GamBinaryTrainer, GamBinaryTrainer.Options>
            Gam(
                this SweepableBinaryClassificationTrainers trainer,
                string labelColumnName = "Label",
                string featureColumnName = "Features",
                SweepableOption<GamBinaryTrainer.Options> optionBuilder = null,
                GamBinaryTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = GamBinaryTrainerSweepableOptions.Default;
            }

            optionBuilder.SetDefaultOption(defaultOption);
            return context.AutoML().CreateSweepableEstimator(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.BinaryClassification.Trainers.Gam(option);
                },
                optionBuilder,
                new string[] { labelColumnName, featureColumnName },
                new string[] { PredictedLabel },
                nameof(GamBinaryTrainer));
        }

        public static SweepableEstimator<SgdNonCalibratedTrainer, SgdNonCalibratedTrainer.Options>
            SgdNonCalibrated(
                this SweepableBinaryClassificationTrainers trainer,
                string labelColumnName = "Label",
                string featureColumnName = "Features",
                SweepableOption<SgdNonCalibratedTrainer.Options> optionBuilder = null,
                SgdNonCalibratedTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = SgdNonCalibratedBinaryTrainerSweepableOptions.Default;
            }

            optionBuilder.SetDefaultOption(defaultOption);
            return context.AutoML().CreateSweepableEstimator(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.BinaryClassification.Trainers.SgdNonCalibrated(option);
                },
                optionBuilder,
                new string[] { labelColumnName, featureColumnName },
                new string[] { PredictedLabel },
                nameof(SgdNonCalibratedTrainer));
        }

        public static SweepableEstimator<SgdCalibratedTrainer, SgdCalibratedTrainer.Options>
            SgdCalibrated(
                this SweepableBinaryClassificationTrainers trainer,
                string labelColumnName = "Label",
                string featureColumnName = "Features",
                SweepableOption<SgdCalibratedTrainer.Options> optionBuilder = null,
                SgdCalibratedTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = SgdCalibratedBinaryTrainerSweepableOptions.Default;
            }

            optionBuilder.SetDefaultOption(defaultOption);
            return context.AutoML().CreateSweepableEstimator(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.BinaryClassification.Trainers.SgdCalibrated(option);
                },
                optionBuilder,
                new string[] { labelColumnName, featureColumnName },
                new string[] { PredictedLabel },
                nameof(SgdCalibratedTrainer));
        }

        public static SweepableEstimator<SdcaNonCalibratedBinaryTrainer, SdcaNonCalibratedBinaryTrainer.Options>
            SdcaNonCalibrated(
                this SweepableBinaryClassificationTrainers trainer,
                string labelColumnName = "Label",
                string featureColumnName = "Features",
                SweepableOption<SdcaNonCalibratedBinaryTrainer.Options> optionBuilder = null,
                SdcaNonCalibratedBinaryTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = SdcaNonCalibratedBinaryTrainerSweepableOptions.Default;
            }

            optionBuilder.SetDefaultOption(defaultOption);
            return context.AutoML().CreateSweepableEstimator(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.BinaryClassification.Trainers.SdcaNonCalibrated(option);
                },
                optionBuilder,
                new string[] { labelColumnName, featureColumnName },
                new string[] { PredictedLabel },
                nameof(SdcaNonCalibratedBinaryTrainer));
        }

        public static SweepableEstimator<SdcaLogisticRegressionBinaryTrainer, SdcaLogisticRegressionBinaryTrainer.Options>
            SdcaLogisticRegression(
                this SweepableBinaryClassificationTrainers trainer,
                string labelColumnName = "Label",
                string featureColumnName = "Features",
                SweepableOption<SdcaLogisticRegressionBinaryTrainer.Options> optionBuilder = null,
                SdcaLogisticRegressionBinaryTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = SdcaLogisticRegressionBinaryTrainerSweepableOptions.Default;
            }

            optionBuilder.SetDefaultOption(defaultOption);
            return context.AutoML().CreateSweepableEstimator(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.BinaryClassification.Trainers.SdcaLogisticRegression(option);
                },
                optionBuilder,
                new string[] { labelColumnName, featureColumnName },
                new string[] { PredictedLabel },
                nameof(SdcaLogisticRegressionBinaryTrainer));
        }

        public static SweepableEstimator<LbfgsLogisticRegressionBinaryTrainer, LbfgsLogisticRegressionBinaryTrainer.Options>
            LbfgsLogisticRegression(
                this SweepableBinaryClassificationTrainers trainer,
                string labelColumnName = "Label",
                string featureColumnName = "Features",
                SweepableOption<LbfgsLogisticRegressionBinaryTrainer.Options> optionBuilder = null,
                LbfgsLogisticRegressionBinaryTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = LbfgsLogisticRegressionBinaryTrainerSweepableOptions.Default;
            }

            optionBuilder.SetDefaultOption(defaultOption);
            return context.AutoML().CreateSweepableEstimator(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.BinaryClassification.Trainers.LbfgsLogisticRegression(option);
                },
                optionBuilder,
                new string[] { labelColumnName, featureColumnName },
                new string[] { PredictedLabel },
                nameof(LbfgsLogisticRegressionBinaryTrainer));
        }

        public static SweepableEstimator<AveragedPerceptronTrainer, AveragedPerceptronTrainer.Options>
            AveragedPerceptron(
                this SweepableBinaryClassificationTrainers trainer,
                string labelColumnName = "Label",
                string featureColumnName = "Features",
                SweepableOption<AveragedPerceptronTrainer.Options> optionBuilder = null,
                AveragedPerceptronTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = AveragedPerceptronBinaryTrainerSweepableOptions.Default;
            }

            optionBuilder.SetDefaultOption(defaultOption);
            return context.AutoML().CreateSweepableEstimator(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.BinaryClassification.Trainers.AveragedPerceptron(option);
                },
                optionBuilder,
                new string[] { labelColumnName, featureColumnName },
                new string[] { PredictedLabel },
                nameof(AveragedPerceptronTrainer));
        }
    }
}
