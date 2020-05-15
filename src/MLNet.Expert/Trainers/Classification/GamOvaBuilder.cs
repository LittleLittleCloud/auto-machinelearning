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
        private Option _option;
        private MLContext _context;
        private static GamOvaBuilder _instance = new GamOvaBuilder();

        public GamOvaBuilder(MLContext context, Option option)
        {
            this._option = option;
            this._context = context;
        }

        internal GamOvaBuilder()
        {
            this._option = new Option();
        }

        internal static GamOvaBuilder Instance
        {
            get => GamOvaBuilder._instance;
        }

        public EstimatorSingleNode CreateTrainer(MLContext context, string label, string feature)
        {
            Func<Option, OneVersusAllTrainer> OVA = (option) =>
            {
                return context.MulticlassClassification.Trainers.OneVersusAll(context.BinaryClassification.Trainers.Gam(label, feature, maximumBinCountPerFeature: option.MaximumBinCountPerFeature, learningRate: option.LearningRate), label);
            };

            var ovaNode = Util.CreateSweepableNode(OVA, this._option, estimatorName: "GamOva");

            return new EstimatorSingleNode(ovaNode);
        }

        /// <summary>
        /// Sweepable option for <see cref="GamOvaBuilder"/>.
        /// </summary>
        public class Option : OptionBuilder<Option>
        {
            [Parameter(1e-4f, 1, true, 20)]
            public float LearningRate;

            [Parameter(4, 512, true, 20)]
            public int MaximumBinCountPerFeature;
        }
    }
}
