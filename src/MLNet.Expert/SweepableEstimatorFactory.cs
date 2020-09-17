// <copyright file="SweepableEstimatorFactory.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Trainers.LightGbm;
using MLNet.AutoPipeline;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.Expert
{
    public static class SweepableEstimatorFactory
    {
        public static SweepableEstimatorBase CreateSweepableEstimator(MLContext context, SweepableEstimatorDataContract estimator)
        {
            switch (estimator.EstimatorName)
            {
                case nameof(LightGbmRegressionTrainer):
                    var label = estimator.InputColumns[0];
                    var feature = estimator.InputColumns[1];
                    return context.AutoML().Regression.LightGbm(label, feature);
                default:
                    throw new Exception($"{estimator.EstimatorName} can't be created through SweepabeEstimatorFactory");
            }
        }
    }
}
