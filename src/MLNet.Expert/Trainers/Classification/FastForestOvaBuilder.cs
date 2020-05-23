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
        private Option _option;
        private MLContext _context;
        private static FastForestOvaBuilder _instance = new FastForestOvaBuilder();

        public FastForestOvaBuilder(MLContext context, Option option)
        {
            this._option = option;
            this._context = context;
        }

        internal FastForestOvaBuilder()
        {
            this._option = new Option();
        }

        internal static FastForestOvaBuilder Instance
        {
            get => FastForestOvaBuilder._instance;
        }

        public EstimatorSingleNode CreateTrainer(MLContext context, string label, string feature)
        {
            Func<Option, OneVersusAllTrainer> OVA = (option) =>
            {
                return context.MulticlassClassification.Trainers.OneVersusAll(context.BinaryClassification.Trainers.FastForest(label, feature, numberOfLeaves: option.NumberOfLeaves, numberOfTrees: option.NumberOfTrees, minimumExampleCountPerLeaf: option.MinimumExampleCountPerLeaf), label);
            };

            var ovaNode = Util.CreateSweepableNode(OVA, this._option, estimatorName: "FastForestOva");

            return new EstimatorSingleNode(ovaNode);
        }

        /// <summary>
        /// Sweepable option for <see cref="FastForestOvaBuilder"/>.
        /// </summary>
        public class Option : OptionBuilder<Option>
        {
            [Parameter(2, 256, true, 8)]
            public int NumberOfLeaves;

            [Parameter(1, 256, true, 20)]
            public int NumberOfTrees;

            [Parameter(1, 256, true, 20)]
            public int MinimumExampleCountPerLeaf;
        }
    }
}
