// <copyright file="LightGBMBuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.LightGbm;
using MLNet.AutoPipeline;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.Expert.Trainers.Classification
{
    /// <summary>
    /// Create a <see cref="Microsoft.ML.Trainers.LightGbm.LightGbmMulticlassTrainer"/> with sweepable option.
    /// </summary>
    public class LightGBMBuilder : ICanCreateTrainer
    {
        private MLContext _context;
        private static LightGBMBuilder _instance = new LightGBMBuilder();

        public LightGBMBuilder(MLContext context)
        {
            this._context = context;
        }

        internal LightGBMBuilder()
        {
        }

        internal static LightGBMBuilder Instance
        {
            get => LightGBMBuilder._instance;
        }

        public EstimatorSingleNode CreateTrainer(MLContext context, string label, string feature)
        {
            var node = context.AutoML().MultiClassification.LightGbm(label, feature);
            return Util.CreateEstimatorSingleNode(node);
        }
    }
}
