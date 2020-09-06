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

        public NaiveBayesBuilder()
        {
        }

        internal static NaiveBayesBuilder Instance
        {
            get => NaiveBayesBuilder.instance;
        }

        public SweepableEstimatorBase CreateTrainer(MLContext context, string label, string feature)
        {
            return context.AutoML().MultiClassification.NaiveBayes(label, feature);
        }
    }
}
