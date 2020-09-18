// <copyright file="SweepableEstimatorFactory.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Transforms;
using MLNet.AutoPipeline;
using MLNet.Expert.Contract;
using MLNet.Expert.Extension;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.Expert
{
    public static class SweepableEstimatorFactory
    {
        public static SweepableEstimatorBase CreateSweepableEstimator(MLContext context, SweepableEstimatorDataContract estimator)
        {
            var input = estimator.InputColumns[0];
            var output = estimator.OutputColumns[0];

            switch (estimator.EstimatorName)
            {
                case nameof(LightGbmRegressionTrainer):
                    var label = estimator.InputColumns[0];
                    var feature = estimator.InputColumns[1];
                    return context.AutoML().Regression.LightGbm(label, feature);
                case nameof(LinearSvmTrainer):
                    label = estimator.InputColumns[0];
                    feature = estimator.InputColumns[1];
                    return context.AutoML().Serializable().BinaryClassification.LinearSvm(label, feature);
                case nameof(LdSvmTrainer):
                    label = estimator.InputColumns[0];
                    feature = estimator.InputColumns[1];
                    return context.AutoML().Serializable().BinaryClassification.LdSvm(label, feature);
                case nameof(FastForestBinaryTrainer):
                    label = estimator.InputColumns[0];
                    feature = estimator.InputColumns[1];
                    return context.AutoML().Serializable().BinaryClassification.FastForest(label, feature);
                case nameof(FastTreeBinaryTrainer):
                    label = estimator.InputColumns[0];
                    feature = estimator.InputColumns[1];
                    return context.AutoML().Serializable().BinaryClassification.FastTree(label, feature);
                case nameof(LightGbmBinaryTrainer):
                    label = estimator.InputColumns[0];
                    feature = estimator.InputColumns[1];
                    return context.AutoML().Serializable().BinaryClassification.LightGbm(label, feature);
                case nameof(GamBinaryTrainer):
                    label = estimator.InputColumns[0];
                    feature = estimator.InputColumns[1];
                    return context.AutoML().Serializable().BinaryClassification.Gam(label, feature);
                case nameof(SgdNonCalibratedTrainer):
                    label = estimator.InputColumns[0];
                    feature = estimator.InputColumns[1];
                    return context.AutoML().Serializable().BinaryClassification.SgdNonCalibrated(label, feature);
                case nameof(SgdCalibratedTrainer):
                    label = estimator.InputColumns[0];
                    feature = estimator.InputColumns[1];
                    return context.AutoML().Serializable().BinaryClassification.SgdCalibrated(label, feature);
                case nameof(AveragedPerceptronTrainer):
                    label = estimator.InputColumns[0];
                    feature = estimator.InputColumns[1];
                    return context.AutoML().Serializable().BinaryClassification.AveragedPerceptron(label, feature);
                case nameof(OneHotEncodingEstimator):
                    return context.AutoML().Serializable().Transformer.Categorical.OneHotEncoding(input, output);
                case nameof(MissingValueReplacingEstimator):
                    return context.AutoML().Serializable().Transformer.ReplaceMissingValues(input, output);
                case nameof(ColumnConcatenatingEstimator):
                    return context.AutoML().Serializable().Transformer.Concatnate(estimator.InputColumns, output);
                default:
                    throw new Exception($"{estimator.EstimatorName} can't be created through SweepabeEstimatorFactory");
            }
        }
    }
}
