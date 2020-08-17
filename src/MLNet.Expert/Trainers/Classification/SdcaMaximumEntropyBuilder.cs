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
        private MLContext _context;
        private static SdcaMaximumEntropyBuilder _instance = new SdcaMaximumEntropyBuilder();

        public SdcaMaximumEntropyBuilder(MLContext context)
        {
            this._context = context;
        }

        internal SdcaMaximumEntropyBuilder()
        {
        }

        internal static SdcaMaximumEntropyBuilder Instance
        {
            get => SdcaMaximumEntropyBuilder._instance;
        }

        public INode CreateTrainer(MLContext context, string label, string feature)
        {
            return context.AutoML().MultiClassification.SdcaMaximumEntropy(label, feature);
        }
    }
}
