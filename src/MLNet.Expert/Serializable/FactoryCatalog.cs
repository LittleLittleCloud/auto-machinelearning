// <copyright file="FactoryCatalog.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;
using MLNet.AutoPipeline;
using MLNet.Expert.Contract;
using MLNet.Sweeper;
using System;
using System.Collections.Generic;
using System.Text;
using static MLNet.Expert.Serializable.SerializableEvaluateFunction;

namespace MLNet.Expert.Serializable
{
    internal class FactoryCatalog
    {
        public FactoryCatalog(MLContext context)
        {
            this.Context = context;
        }

        public MLContext Context { get; }

        public SweepableEstimatorBase CreateSweepableEstimator(SweepableEstimatorDataContract estimator)
        {
            var input = estimator.InputColumns[0];
            var output = estimator.OutputColumns[0];

            switch (estimator.EstimatorName)
            {
                case nameof(LightGbmRegressionTrainer):
                    var label = estimator.InputColumns[0];
                    var feature = estimator.InputColumns[1];
                    return this.Context.AutoML().Regression.LightGbm(label, feature);
                case nameof(LinearSvmTrainer):
                    label = estimator.InputColumns[0];
                    feature = estimator.InputColumns[1];
                    return this.Context.AutoML().Serializable().BinaryClassification.LinearSvm(label, feature);
                case nameof(LdSvmTrainer):
                    label = estimator.InputColumns[0];
                    feature = estimator.InputColumns[1];
                    return this.Context.AutoML().Serializable().BinaryClassification.LdSvm(label, feature);
                case nameof(FastForestBinaryTrainer):
                    label = estimator.InputColumns[0];
                    feature = estimator.InputColumns[1];
                    return this.Context.AutoML().Serializable().BinaryClassification.FastForest(label, feature);
                case nameof(FastTreeBinaryTrainer):
                    label = estimator.InputColumns[0];
                    feature = estimator.InputColumns[1];
                    return this.Context.AutoML().Serializable().BinaryClassification.FastTree(label, feature);
                case nameof(LightGbmBinaryTrainer):
                    label = estimator.InputColumns[0];
                    feature = estimator.InputColumns[1];
                    return this.Context.AutoML().Serializable().BinaryClassification.LightGbm(label, feature);
                case nameof(GamBinaryTrainer):
                    label = estimator.InputColumns[0];
                    feature = estimator.InputColumns[1];
                    return this.Context.AutoML().Serializable().BinaryClassification.Gam(label, feature);
                case nameof(SgdNonCalibratedTrainer):
                    label = estimator.InputColumns[0];
                    feature = estimator.InputColumns[1];
                    return this.Context.AutoML().Serializable().BinaryClassification.SgdNonCalibrated(label, feature);
                case nameof(SgdCalibratedTrainer):
                    label = estimator.InputColumns[0];
                    feature = estimator.InputColumns[1];
                    return this.Context.AutoML().Serializable().BinaryClassification.SgdCalibrated(label, feature);
                case nameof(AveragedPerceptronTrainer):
                    label = estimator.InputColumns[0];
                    feature = estimator.InputColumns[1];
                    return this.Context.AutoML().Serializable().BinaryClassification.AveragedPerceptron(label, feature);
                case nameof(OneHotEncodingEstimator):
                    return this.Context.AutoML().Serializable().Transformer.Categorical.OneHotEncoding(input, output);
                case nameof(MissingValueReplacingEstimator):
                    return this.Context.AutoML().Serializable().Transformer.ReplaceMissingValues(input, output);
                case nameof(ColumnConcatenatingEstimator):
                    return this.Context.AutoML().Serializable().Transformer.Concatnate(estimator.InputColumns, output);
                case nameof(TextFeaturizingEstimator):
                    return this.Context.AutoML().Serializable().Transformer.Text.FeaturizeText(input, output);
                case nameof(SerializableTextCatalog.FeaturizeTextWithWordEmbedding):
                    return this.Context.AutoML().Serializable().Transformer.Text.FeaturizeTextWithWordEmbedding(input, output);
                default:
                    throw new Exception($"{estimator.EstimatorName} can't be created through SweepabeEstimatorFactory");
            }
        }

        public ISweeper CreateSweeper(string name)
        {
            switch (name)
            {
                case nameof(GridSearchSweeper):
                    return this.Context.AutoML().Serializable().Sweeper.CreateGridSearchSweeper();
                case nameof(RandomGridSweeper):
                    return this.Context.AutoML().Serializable().Sweeper.CreateRandomGridSweeper();
                default:
                    throw new Exception($"{name} can't be created through SweeperFactory");
            }
        }

        public EvaluateFunctionWithLabel CreateEvaluateFunction(string name)
        {
            switch (name)
            {
                // TODO
                // use reflection
                case nameof(SerializableEvaluateFunction.Accuracy):
                    return this.Context.AutoML().Serializable().EvaluationFunction.Accuracy;
                default:
                    throw new Exception($"{name} can't be created through SerializableEvaluateFunction");
            };
        }
    }
}
