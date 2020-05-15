// <copyright file="NaiveBayesBuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using MLNet.AutoPipeline;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace MLNet.Expert.Trainers.Classification
{
    /// <summary>
    /// Create a <see cref="Microsoft.ML.Trainers.NaiveBayesMulticlassTrainer"/> with sweepable option.
    /// </summary>
    public class NaiveBayesBuilder : ICanCreateTrainer
    {
        private static NaiveBayesBuilder instance = new NaiveBayesBuilder();
        private Option _option;

        public NaiveBayesBuilder(Option option)
        {
            this._option = option;
        }

        internal NaiveBayesBuilder()
        {
            this._option = new Option();
        }

        internal static NaiveBayesBuilder Instance
        {
            get => NaiveBayesBuilder.instance;
        }

        public EstimatorSingleNode CreateTrainer(MLContext context, string label, string feature)
        {
            var instance = context.MulticlassClassification.Trainers.NaiveBayes(label, feature);
            var pipelineNode = new UnsweepableNode<MulticlassPredictionTransformer<NaiveBayesMulticlassModelParameters>>(instance);
            return new EstimatorSingleNode(pipelineNode);
        }

        public class Option { }
    }
}
