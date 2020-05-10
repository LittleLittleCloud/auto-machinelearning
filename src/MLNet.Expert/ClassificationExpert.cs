// <copyright file="ClassificationExpert.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using MLNet.AutoPipeline;
using MLNet.Expert.Trainers.Classification;
using System.Collections;
using System.Collections.Generic;

namespace MLNet.Expert
{
    public class ClassificationExpert : ITrainerExpert
    {
        private IList<ICanCreateTrainer> nodeFactories = new List<ICanCreateTrainer>();
        private MLContext context;

        public ClassificationExpert(MLContext context, Option option)
        {
            this.context = context;
            if (option.UseNaiveBayes)
            {
                this.nodeFactories.Add(NaiveBayesBuilder.Instance);
            }

            if (option.UseLbfgsMaximumEntropy)
            {
                this.nodeFactories.Add(LbfgsMaximumEntropyBuilder.Instance);
            }
        }

        public IEstimatorNode Propose(string label, string feature)
        {
            var groupNode = new EstimatorNodeGroup();
            foreach (var creator in this.nodeFactories)
            {
                groupNode.Append(creator.CreateTrainer(this.context, label, feature));
            }

            return groupNode;
        }

        public class Option
        {
            /// <summary>
            /// Use NaiveBayes classifier created by <see cref="NaiveBayesBuilder"/>.
            /// </summary>
            public bool UseNaiveBayes { get; set; } = true;

            /// <summary>
            /// Use UseLbfgsMaximumEntropy classifier created by <see cref="LbfgsMaximumEntropyBuilder"/>.
            /// </summary>
            public bool UseLbfgsMaximumEntropy { get; set; } = true;
        }
    }
}
