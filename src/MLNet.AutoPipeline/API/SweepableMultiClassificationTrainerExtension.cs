// <copyright file="SweepableMultiClassificationTrainerExtension.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Reflection.Emit;
using System.Text;

namespace MLNet.AutoPipeline
{
    /// <summary>
    /// class contains extension method for <see cref="SweepableMultiClassificationTrainers"/>.
    /// </summary>
    public static class SweepableMultiClassificationTrainerExtension
    {
        /// <summary>
        /// Create an <see cref="INode"/> of <see cref="NaiveBayesMulticlassTrainer"/> that can be used in <see cref="ISweepablePipeline"/>. Notice from the fact that there is no sweepable parameters for Naive Bayes, So the return type is <see cref=" UnsweepableNode{TTransformer}"/>.
        /// </summary>
        /// <param name="trainer">The <see cref=" SweepableMultiClassificationTrainerExtension"/>.</param>
        /// <param name="labelColumnName">label column name. Default is Label.</param>
        /// <param name="featureColumnName">feature column name. Default is Features</param>
        /// <returns><see cref="UnsweepableNode{TTransformer}"/> where TTransformer is <see cref="NaiveBayesMulticlassTrainer"/>.</returns>
        public static UnsweepableNode<NaiveBayesMulticlassTrainer> NaiveBayes(this SweepableMultiClassificationTrainers trainer, string labelColumnName = "Label", string featureColumnName = "Features")
        {
            var context = trainer.Context;
            var instance = context.MulticlassClassification.Trainers.NaiveBayes(labelColumnName, featureColumnName);
            return Util.CreateUnSweepableNode(
                        instance,
                        estimatorName: nameof(NaiveBayesMulticlassTrainer),
                        inputs: new string[] { labelColumnName },
                        outputs: new string[] { featureColumnName });
        }
    }
}
