// <copyright file="SdcaNonCalibratedBuilder.cs" company="BigMiao">
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
    /// Create a <see cref="Microsoft.ML.Trainers.SdcaNonCalibratedMulticlassTrainer"/> with sweepable option.
    /// </summary>
    public class SdcaNonCalibratedBuilder : ICanCreateTrainer
    {
        private MLContext _context;
        private static SdcaNonCalibratedBuilder _instance = new SdcaNonCalibratedBuilder();

        public SdcaNonCalibratedBuilder(MLContext context)
        {
            this._context = context;
        }

        internal SdcaNonCalibratedBuilder()
        {
        }

        internal static SdcaNonCalibratedBuilder Instance
        {
            get => SdcaNonCalibratedBuilder._instance;
        }

        public EstimatorSingleNode CreateTrainer(MLContext context, string label, string feature)
        {
            var node = context.AutoML().MultiClassification.SdcaNonCalibreated(label, feature);
            return Util.CreateEstimatorSingleNode(node);
        }
    }
}
