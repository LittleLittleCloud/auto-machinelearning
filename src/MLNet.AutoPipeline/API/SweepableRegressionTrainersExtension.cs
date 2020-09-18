// <copyright file="SweepableRegressionTrainersExtension.cs" company="BigMiao">
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
    public static class SweepableRegressionTrainersExtension
    {
        internal const string Score = "Score";

        public static SweepableEstimator<LightGbmRegressionTrainer, LightGbmRegressionTrainer.Options>
            LightGbm(
                this SweepableRegressionTrainers trainers,
                string labelColumnName = "Label",
                string featureColumnName = "Features",
                SweepableOption<LightGbmRegressionTrainer.Options> optionSweeper = null,
                LightGbmRegressionTrainer.Options defaultOption = null)
        {
            var context = trainers.Context;

            if (optionSweeper == null)
            {
                optionSweeper = LightGbmRegressionTrainerSweepableOptions.Default;
            }

            optionSweeper.SetDefaultOption(defaultOption);

            return context.AutoML().CreateSweepableEstimator(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.Regression.Trainers.LightGbm(option);
                },
                optionSweeper,
                new string[] { labelColumnName, featureColumnName },
                new string[] { Score },
                nameof(LightGbmRegressionTrainer));
        }

        public static SweepableEstimator<LbfgsPoissonRegressionTrainer, LbfgsPoissonRegressionTrainer.Options>
            LbfgsPoissonRegression(
                this SweepableRegressionTrainers trainers,
                string labelColumnName = "Label",
                string featureColumnName = "Features",
                SweepableOption<LbfgsPoissonRegressionTrainer.Options> optionSweeper = null,
                LbfgsPoissonRegressionTrainer.Options defaultOption = null)
        {
            var context = trainers.Context;

            if (optionSweeper == null)
            {
                optionSweeper = LbfgsPoissonRegressionTrainerSweepableOptions.Default;
            }

            optionSweeper.SetDefaultOption(defaultOption);

            return context.AutoML().CreateSweepableEstimator(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.Regression.Trainers.LbfgsPoissonRegression(option);
                },
                optionSweeper,
                new string[] { labelColumnName, featureColumnName },
                new string[] { Score },
                nameof(LbfgsPoissonRegressionTrainer));
        }

        public static SweepableEstimator<OnlineGradientDescentTrainer, OnlineGradientDescentTrainer.Options>
            OnlineGradientDescent(
                this SweepableRegressionTrainers trainers,
                string labelColumnName = "Label",
                string featureColumnName = "Features",
                SweepableOption<OnlineGradientDescentTrainer.Options> optionSweeper = null,
                OnlineGradientDescentTrainer.Options defaultOption = null)
        {
            var context = trainers.Context;

            if (optionSweeper == null)
            {
                optionSweeper = OnlineGradientDescentTrainerSweepableOptions.Default;
            }

            optionSweeper.SetDefaultOption(defaultOption);

            return context.AutoML().CreateSweepableEstimator(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.Regression.Trainers.OnlineGradientDescent(option);
                },
                optionSweeper,
                new string[] { labelColumnName, featureColumnName },
                new string[] { Score },
                nameof(OnlineGradientDescentTrainer));
        }

        public static SweepableEstimator<SdcaRegressionTrainer, SdcaRegressionTrainer.Options>
            Sdca(
                this SweepableRegressionTrainers trainers,
                string labelColumnName = "Label",
                string featureColumnName = "Features",
                SweepableOption<SdcaRegressionTrainer.Options> optionSweeper = null,
                SdcaRegressionTrainer.Options defaultOption = null)
        {
            var context = trainers.Context;

            if (optionSweeper == null)
            {
                optionSweeper = SdcaRegressionTrainerSweepableOptions.Default;
            }

            optionSweeper.SetDefaultOption(defaultOption);

            return context.AutoML().CreateSweepableEstimator(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.Regression.Trainers.Sdca(option);
                },
                optionSweeper,
                new string[] { labelColumnName, featureColumnName },
                new string[] { Score },
                nameof(SdcaRegressionTrainer));
        }

        public static SweepableEstimator<FastForestRegressionTrainer, FastForestRegressionTrainer.Options>
            FastForest(
                this SweepableRegressionTrainers trainers,
                string labelColumnName = "Label",
                string featureColumnName = "Features",
                SweepableOption<FastForestRegressionTrainer.Options> optionSweeper = null,
                FastForestRegressionTrainer.Options defaultOption = null)
        {
            var context = trainers.Context;

            if (optionSweeper == null)
            {
                optionSweeper = FastForestRegressionTrainerSweepableOptions.Default;
            }

            optionSweeper.SetDefaultOption(defaultOption);

            return context.AutoML().CreateSweepableEstimator(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.Regression.Trainers.FastForest(option);
                },
                optionSweeper,
                new string[] { labelColumnName, featureColumnName },
                new string[] { Score },
                nameof(FastForestRegressionTrainer));
        }

        public static SweepableEstimator<FastTreeRegressionTrainer, FastTreeRegressionTrainer.Options>
            FastTree(
                this SweepableRegressionTrainers trainers,
                string labelColumnName = "Label",
                string featureColumnName = "Features",
                SweepableOption<FastTreeRegressionTrainer.Options> optionSweeper = null,
                FastTreeRegressionTrainer.Options defaultOption = null)
        {
            var context = trainers.Context;

            if (optionSweeper == null)
            {
                optionSweeper = FastTreeRegressionTrainerSweepableOptions.Default;
            }

            optionSweeper.SetDefaultOption(defaultOption);

            return context.AutoML().CreateSweepableEstimator(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.Regression.Trainers.FastTree(option);
                },
                optionSweeper,
                new string[] { labelColumnName, featureColumnName },
                new string[] { Score },
                nameof(FastTreeRegressionTrainer));
        }

        public static SweepableEstimator<FastTreeTweedieTrainer, FastTreeTweedieTrainer.Options>
            FastTreeTweedie(
                this SweepableRegressionTrainers trainers,
                string labelColumnName = "Label",
                string featureColumnName = "Features",
                SweepableOption<FastTreeTweedieTrainer.Options> optionSweeper = null,
                FastTreeTweedieTrainer.Options defaultOption = null)
        {
            var context = trainers.Context;

            if (optionSweeper == null)
            {
                optionSweeper = FastTreeTweedieTrainerSweepableOptions.Default;
            }

            optionSweeper.SetDefaultOption(defaultOption);

            return context.AutoML().CreateSweepableEstimator(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.Regression.Trainers.FastTreeTweedie(option);
                },
                optionSweeper,
                new string[] { labelColumnName, featureColumnName },
                new string[] { Score },
                nameof(FastTreeTweedieTrainer));
        }

        public static SweepableEstimator<GamRegressionTrainer, GamRegressionTrainer.Options>
            Gam(
                this SweepableRegressionTrainers trainers,
                string labelColumnName = "Label",
                string featureColumnName = "Features",
                SweepableOption<GamRegressionTrainer.Options> optionSweeper = null,
                GamRegressionTrainer.Options defaultOption = null)
        {
            var context = trainers.Context;

            if (optionSweeper == null)
            {
                optionSweeper = GamRegressionTrainerSweepableOptions.Default;
            }

            optionSweeper.SetDefaultOption(defaultOption);

            return context.AutoML().CreateSweepableEstimator(
                (context, option) =>
                {
                    option.LabelColumnName = labelColumnName;
                    option.FeatureColumnName = featureColumnName;

                    return context.Regression.Trainers.Gam(option);
                },
                optionSweeper,
                new string[] { labelColumnName, featureColumnName },
                new string[] { Score },
                nameof(GamRegressionTrainer));
        }
    }
}
