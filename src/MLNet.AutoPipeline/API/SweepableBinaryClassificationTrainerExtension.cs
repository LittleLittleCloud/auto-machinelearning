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
        private static string PredictedLabel = "PredictedLabel";

        public static SweepableNode<LinearSvmTrainer, LinearSvmTrainer.Options>
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
                SweepableOption<LdSvmTrainer.Options> optionBuilder = null,
                LdSvmTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = LdSvmBinaryTrainerSweepableOptions.Default;
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
                SweepableOption<FastForestBinaryTrainer.Options> optionBuilder = null,
                FastForestBinaryTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = FastForestBinaryTrainerSweepableOptions.Default;
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
                SweepableOption<FastTreeBinaryTrainer.Options> optionBuilder = null,
                FastTreeBinaryTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = FastTreeBinaryTrainerSweepableOptions.Default;
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
                SweepableOption<LightGbmBinaryTrainer.Options> optionBuilder = null,
                LightGbmBinaryTrainer.Options defaultOption = null)
        {
            var context = trainer.Context;
            if (optionBuilder == null)
            {
                optionBuilder = LightGbmBinaryTrainerSweepableOptions.Default;
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
                nameof(LightGbmBinaryTrainer));
        }

        public static SweepableNode<GamBinaryTrainer, GamBinaryTrainer.Options>
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
            return context.AutoML().SweepableTrainer(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.BinaryClassification.Trainers.Gam(option);
                },
                optionBuilder,
                new string[] { featureColumnName },
                PredictedLabel,
                nameof(GamBinaryTrainer));
        }

        public static SweepableNode<SgdNonCalibratedTrainer, SgdNonCalibratedTrainer.Options>
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
            return context.AutoML().SweepableTrainer(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.BinaryClassification.Trainers.SgdNonCalibrated(option);
                },
                optionBuilder,
                new string[] { featureColumnName },
                PredictedLabel,
                nameof(SgdNonCalibratedTrainer));
        }

        public static SweepableNode<SgdCalibratedTrainer, SgdCalibratedTrainer.Options>
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
            return context.AutoML().SweepableTrainer(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.BinaryClassification.Trainers.SgdCalibrated(option);
                },
                optionBuilder,
                new string[] { featureColumnName },
                PredictedLabel,
                nameof(SgdCalibratedTrainer));
        }

        public static SweepableNode<SdcaNonCalibratedBinaryTrainer, SdcaNonCalibratedBinaryTrainer.Options>
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
            return context.AutoML().SweepableTrainer(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.BinaryClassification.Trainers.SdcaNonCalibrated(option);
                },
                optionBuilder,
                new string[] { featureColumnName },
                PredictedLabel,
                nameof(SdcaNonCalibratedBinaryTrainer));
        }

        public static SweepableNode<SdcaLogisticRegressionBinaryTrainer, SdcaLogisticRegressionBinaryTrainer.Options>
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
            return context.AutoML().SweepableTrainer(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.BinaryClassification.Trainers.SdcaLogisticRegression(option);
                },
                optionBuilder,
                new string[] { featureColumnName },
                PredictedLabel,
                nameof(SdcaLogisticRegressionBinaryTrainer));
        }

        public static SweepableNode<LbfgsLogisticRegressionBinaryTrainer, LbfgsLogisticRegressionBinaryTrainer.Options>
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
            return context.AutoML().SweepableTrainer(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.BinaryClassification.Trainers.LbfgsLogisticRegression(option);
                },
                optionBuilder,
                new string[] { featureColumnName },
                PredictedLabel,
                nameof(LbfgsLogisticRegressionBinaryTrainer));
        }

        public static SweepableNode<AveragedPerceptronTrainer, AveragedPerceptronTrainer.Options>
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
            return context.AutoML().SweepableTrainer(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.BinaryClassification.Trainers.AveragedPerceptron(option);
                },
                optionBuilder,
                new string[] { featureColumnName },
                PredictedLabel,
                nameof(AveragedPerceptronTrainer));
        }
    }
}
