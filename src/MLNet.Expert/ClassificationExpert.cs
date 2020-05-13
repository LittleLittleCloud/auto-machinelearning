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

            if (option.UseNaiveBayes)
            {
                this.nodeFactories.Add(LightGBMBuilder.Instance);
            }

            if (option.UseSdcaMaximumEntropy)
            {
                this.nodeFactories.Add(SdcaMaximumEntropyBuilder.Instance);
            }

            if (option.UseSdcaNonCalibrated)
            {
                this.nodeFactories.Add(SdcaNonCalibratedBuilder.Instance);
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
            /// Whether to use <see cref="Microsoft.ML.Trainers.NaiveBayesMulticlassTrainer"/> created by <see cref="NaiveBayesBuilder"/>.
            /// </summary>
            public bool UseNaiveBayes { get; set; } = true;

            /// <summary>
            /// Whether to use <see cref="Microsoft.ML.Trainers.LbfgsMaximumEntropyMulticlassTrainer"/> created by <see cref="LbfgsMaximumEntropyBuilder"/>.
            /// </summary>
            public bool UseLbfgsMaximumEntropy { get; set; } = true;

            /// <summary>
            /// Whether to use <see cref="Microsoft.ML.Trainers.LightGbm.LightGbmMulticlassTrainer"/> created by <see cref="LightGBMBuilder"/>.
            /// </summary>
            public bool UseLightGBM { get; set; } = true;

            /// <summary>
            /// Whether to use <see cref="Microsoft.ML.Trainers.SdcaMaximumEntropyMulticlassTrainer"/> created by <see cref="SdcaMaximumEntropyBuilder"/>.
            /// </summary>
            public bool UseSdcaMaximumEntropy { get; set; } = true;

            /// <summary>
            /// Whether to use <see cref="Microsoft.ML.Trainers.SdcaNonCalibratedMulticlassTrainer"/> created by <see cref="SdcaNonCalibratedBuilder"/>.
            /// </summary>
            public bool UseSdcaNonCalibrated { get; set; } = true;
        }
    }
}
