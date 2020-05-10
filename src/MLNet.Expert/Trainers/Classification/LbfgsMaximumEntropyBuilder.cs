// <copyright file="LbfgsMaximumEntropyBuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using MLNet.AutoPipeline;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.Expert.Trainers.Classification
{
    /// <summary>
    /// Create a <see cref="Microsoft.ML.Trainers.LbfgsMaximumEntropyMulticlassTrainer"/> with sweepable <see cref="Option>."/>.
    /// </summary>
    public class LbfgsMaximumEntropyBuilder : ICanCreateTrainer
    {
        private static LbfgsMaximumEntropyBuilder instance = new LbfgsMaximumEntropyBuilder();
        private Option _option;
        private MLContext context;

        public LbfgsMaximumEntropyBuilder(MLContext context, Option option)
        {
            this.context = context;
            this._option = option;
        }

        internal LbfgsMaximumEntropyBuilder()
        {
            this._option = new Option();
        }

        internal static LbfgsMaximumEntropyBuilder Instance
        {
            get => LbfgsMaximumEntropyBuilder.instance;
        }

        public EstimatorSingleNode CreateTrainer(MLContext context, string label, string feature)
        {
            var sweepableNode = new SweepableNode<MulticlassPredictionTransformer<MaximumEntropyModelParameters>, Option>(
                                (option) =>
                                {
                                    return context.MulticlassClassification.Trainers.LbfgsMaximumEntropy(label, feature, null, option.L1Regularization, option.L2Regularization, historySize: option.HistorySize);
                                },
                                this._option,
                                estimatorName: "LbfgsMaximumEntropy");

            return new EstimatorSingleNode(sweepableNode);
        }

        /// <summary>
        /// Sweepable option for <see cref="LbfgsMaximumEntropyBuilder"/>.
        /// </summary>
        public class Option : OptionBuilder<Option>
        {
            /// <summary>
            /// The L1 regularization hyperparameter. Higher values will tend to lead to more sparse model.
            /// </summary>
            [Parameter(1 / 32, 32f, true, 10)]
            public float L1Regularization;

            /// <summary>
            /// The L2 weight for regularization.
            /// </summary>
            [Parameter(1 / 32, 32f, true, 10)]
            public float L2Regularization;

            /// <summary>
            /// Memory size for <see cref="Microsoft.ML.Trainers.LbfgsMaximumEntropyMulticlassTrainer"/>. Low=faster, less accurate.
            /// </summary>
            [Parameter(1, 128, true, 7)]
            public int HistorySize;
        }
    }
}
