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

            if (option.UseFastForestOva)
            {
                this.nodeFactories.Add(Trainers.Classification.FastForestOvaBuilder.Instance);
            }

            if (option.UseFastTreeOva)
            {
                this.nodeFactories.Add(FastTreeOvaBuilder.Instance);
            }

            if (option.UseGamOva)
            {
                this.nodeFactories.Add(GamOvaBuilder.Instance);
            }
        }

        public IEnumerable<SweepableEstimatorBase> Propose(string label, string feature)
        {
            var nodes = new List<SweepableEstimatorBase>();
            foreach (var creator in this.nodeFactories)
            {
                nodes.Add(creator.CreateTrainer(this.context, label, feature));
            }

            return nodes;
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

            /// <summary>
            /// Whether to use FastForest created by <see cref="FastForestOvaBuilder"/>.
            /// </summary>
            public bool UseFastForestOva { get; set; } = true;

            /// <summary>
            /// Whether to use FastTree created by <see cref="FastTreeOvaBuilder"/>.
            /// </summary>
            public bool UseFastTreeOva { get; set; } = true;


            /// <summary>
            /// Whether to use Gam created by <see cref="GamOvaBuilder"/>.
            /// </summary>
            public bool UseGamOva { get; set; } = true;
        }
    }
}
