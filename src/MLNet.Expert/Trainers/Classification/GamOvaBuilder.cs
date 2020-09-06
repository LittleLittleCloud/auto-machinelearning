// <copyright file="FastForestOVABuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using MLNet.AutoPipeline;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.Expert.Trainers.Classification
{
    /// <summary>
    /// Create a Gam multi-classifier using <see cref="Microsoft.ML.Trainers.FastTree.GamBinaryTrainer"/> and <see cref="OneVersusAllTrainer"/> with sweepable option.
    /// </summary>
    public class GamOvaBuilder : ICanCreateTrainer
    {
        private MLContext _context;
        private static GamOvaBuilder _instance = new GamOvaBuilder();

        public GamOvaBuilder(MLContext context)
        {
            this._context = context;
        }

        internal GamOvaBuilder()
        {
        }

        internal static GamOvaBuilder Instance
        {
            get => GamOvaBuilder._instance;
        }

        public SweepableEstimatorBase CreateTrainer(MLContext context, string label, string feature)
        {
            return context.AutoML().MultiClassification.OneVersusAll(context.AutoML().BinaryClassification.Gam(label, feature), label);
        }
    }
}
