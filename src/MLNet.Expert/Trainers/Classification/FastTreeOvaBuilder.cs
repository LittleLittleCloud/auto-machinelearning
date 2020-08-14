// <copyright file="FastTreeOvaBuilder.cs" company="BigMiao">
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
    /// Create a FastTree multi-classifier using <see cref="Microsoft.ML.Trainers.FastTree.FastTreeBinaryTrainer"/> and <see cref="OneVersusAllTrainer"/> with sweepable option.
    /// </summary>
    public class FastTreeOvaBuilder : ICanCreateTrainer
    {
        private MLContext _context;
        private static FastTreeOvaBuilder _instance = new FastTreeOvaBuilder();

        public FastTreeOvaBuilder(MLContext context)
        {
            this._context = context;
        }

        internal FastTreeOvaBuilder()
        {
        }

        internal static FastTreeOvaBuilder Instance
        {
            get => FastTreeOvaBuilder._instance;
        }

        public EstimatorSingleNode CreateTrainer(MLContext context, string label, string feature)
        {
            var trainer = context.AutoML().MultiClassification.OneVersusAll(context.AutoML().BinaryClassification.FastForest(label, feature), label);
            return Util.CreateEstimatorSingleNode(trainer);
        }
    }
}
