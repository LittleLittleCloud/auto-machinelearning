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
    /// Create a FastForest multi-classifier using <see cref="Microsoft.ML.Trainers.FastTree.FastForestBinaryTrainer"/> and <see cref="OneVersusAllTrainer"/> with sweepable option.
    /// </summary>
    public class FastForestOvaBuilder : ICanCreateTrainer
    {
        private MLContext _context;
        private static FastForestOvaBuilder _instance = new FastForestOvaBuilder();

        public FastForestOvaBuilder(MLContext context)
        {
            this._context = context;
        }

        internal FastForestOvaBuilder()
        {
        }

        internal static FastForestOvaBuilder Instance
        {
            get => FastForestOvaBuilder._instance;
        }

        public SweepableEstimatorBase CreateTrainer(MLContext context, string label, string feature)
        {
            var ovaNode = context.AutoML().MultiClassification.OneVersusAll(context.AutoML().BinaryClassification.FastForest(label, feature), label);
            return ovaNode;
        }
    }
}
