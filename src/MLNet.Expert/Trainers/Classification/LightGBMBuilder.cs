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
        private Option _option;
        private MLContext _context;
        private static LightGBMBuilder _instance = new LightGBMBuilder();

        public LightGBMBuilder(MLContext context, Option option)
        {
            this._option = option;
            this._context = context;
        }

        internal LightGBMBuilder()
        {
            this._option = new Option();
        }

        internal static LightGBMBuilder Instance
        {
            get => LightGBMBuilder._instance;
        }

        public EstimatorSingleNode CreateTrainer(MLContext context, string label, string feature)
        {
            Func<Option, LightGbmMulticlassTrainer> lightGBM = (option) =>
            {
                return context.MulticlassClassification.Trainers.LightGbm(label, feature, learningRate: option.LearningRate, numberOfLeaves: option.NumberOfLeaves, minimumExampleCountPerLeaf: option.MinimumExampleCountPerLeaf);
            };

            var node = new SweepableNode<MulticlassPredictionTransformer<OneVersusAllModelParameters>, Option>(lightGBM, this._option, estimatorName: "LightGBM");
            return new EstimatorSingleNode(node);
        }

        /// <summary>
        /// Sweepable option for <see cref="LightGBMBuilder"/>.
        /// </summary>
        public class Option : OptionBuilder<Option>
        {
            [Parameter(0.001f, 0.1f, true, 20)]
            public float LearningRate;

            [Parameter(10, 1000, true, 20)]
            public int NumberOfLeaves;

            [Parameter(10, 1000, true, 20)]
            public int NumberOfIterations;

            [Parameter(10, 1000, true, 20)]
            public int MinimumExampleCountPerLeaf;
        }
    }
}
