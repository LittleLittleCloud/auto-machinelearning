// <copyright file="NumericFeatureExpert.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Transforms;
using MLNet.AutoPipeline;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.Expert
{
    /// <summary>
    /// Expert for numeric feature.
    /// </summary>
    public class NumericFeatureExpert : ITransformExpert
    {
        private MLContext context;
        private Option option;
        private IList<CreateNormalizer> NormalizerFactory;

        private delegate EstimatorSingleNode CreateNormalizer(MLContext context, string input, string output);

        public NumericFeatureExpert(MLContext context)
            : this(context, new Option())
        {
        }

        public static NumericFeatureExpert GetDefaultNumericFeatureExpert(MLContext mLContext)
        {
            return new NumericFeatureExpert(mLContext);
        }

        public NumericFeatureExpert(MLContext context, Option option)
        {
            this.context = context;
            this.option = option;
            this.NormalizerFactory = new List<CreateNormalizer>();

            if (option.NormalizeMeanVariance)
            {
                this.NormalizerFactory.Add(this.CreateNormalizeMeanVariance);
            }

            if (option.NormalizeMinMax)
            {
                this.NormalizerFactory.Add(this.CreateNormalizeMinMax);
            }
        }

        public IEstimatorNode Propose(string inputColumn, string outputColumn)
        {
            var groupNode = new EstimatorNodeGroup();
            foreach ( var creator in this.NormalizerFactory)
            {
                groupNode.Append(creator(this.context, inputColumn, outputColumn));
            }

            return groupNode;
        }

        public IEstimatorNode Propose(string inputColumn)
        {
            return this.Propose(inputColumn, inputColumn);
        }

        public class Option
        {
            /// <summary>
            /// Normalizes based on the computed mean and variance of data.
            /// </summary>
            public bool NormalizeMeanVariance { get; set; } = true;

            /// <summary>
            /// Normalizes the observed minimum and maximum values of data.
            /// </summary>
            public bool NormalizeMinMax { get; set; } = true;
        }

        private EstimatorSingleNode CreateNormalizeMeanVariance(MLContext context, string input, string output)
        {
            var instance = context.Transforms.NormalizeMeanVariance(output, input);
            var unsweeplableNode = new UnsweepableNode<NormalizingTransformer>(instance, estimatorName: "NormalizeMeanVariance");
            return new EstimatorSingleNode(unsweeplableNode);
        }

        private EstimatorSingleNode CreateNormalizeMinMax(MLContext context, string input, string output)
        {
            var instance = context.Transforms.NormalizeMinMax(output, input);
            var unsweeplableNode = new UnsweepableNode<NormalizingTransformer>(instance, estimatorName: "NormalizeMinMax");
            return new EstimatorSingleNode(unsweeplableNode);
        }
    }
}
