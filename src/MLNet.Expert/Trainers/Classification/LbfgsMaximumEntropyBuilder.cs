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
        private MLContext context;

        public LbfgsMaximumEntropyBuilder(MLContext context)
        {
            this.context = context;
        }

        internal LbfgsMaximumEntropyBuilder()
        {
        }

        internal static LbfgsMaximumEntropyBuilder Instance
        {
            get => LbfgsMaximumEntropyBuilder.instance;
        }

        public EstimatorSingleNode CreateTrainer(MLContext context, string label, string feature)
        {
            var sweepableNode = context.AutoML().MultiClassification.LbfgsMaximumEntropy(label, feature);

            return Util.CreateEstimatorSingleNode(sweepableNode);
        }
    }
}
