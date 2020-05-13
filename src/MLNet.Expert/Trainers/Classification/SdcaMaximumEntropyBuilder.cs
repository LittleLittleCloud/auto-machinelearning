// <copyright file="SdcaMaximumEntropyBuilder.cs" company="BigMiao">
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
    /// Create a <see cref="Microsoft.ML.Trainers.SdcaMaximumEntropyMulticlassTrainer"/> with sweepable option.
    /// </summary>
    public class SdcaMaximumEntropyBuilder : ICanCreateTrainer
    {
        private Option _option;
        private MLContext _context;
        private static SdcaMaximumEntropyBuilder _instance = new SdcaMaximumEntropyBuilder();

        public SdcaMaximumEntropyBuilder(MLContext context, Option option)
        {
            this._context = context;
            this._option = option;
        }

        internal SdcaMaximumEntropyBuilder()
        {
            this._option = new Option();
        }

        internal static SdcaMaximumEntropyBuilder Instance
        {
            get => SdcaMaximumEntropyBuilder._instance;
        }

        public EstimatorSingleNode CreateTrainer(MLContext context, string label, string feature)
        {
            Func<Option, SdcaMaximumEntropyMulticlassTrainer> sdca = (option) =>
            {
                return context.MulticlassClassification.Trainers.SdcaMaximumEntropy(label, feature, l1Regularization: option.L1Reegularization, l2Regularization: option.L2Regularization);
            };

            var node = Util.CreateSweepableNode(sdca, this._option, estimatorName: "SdcaMaximumEntropy");
            return new EstimatorSingleNode(node);
        }

        /// <summary>
        /// Sweepable option for <see cref="SdcaMaximumEntropyBuilder"/>.
        /// </summary>
        public class Option : OptionBuilder<Option>
        {
            [Parameter(1E-4F, 10f, true, 20)]
            public float L2Regularization;

            [Parameter(1E-4F, 10f, true, 20)]
            public float L1Reegularization;
        }
    }
}
