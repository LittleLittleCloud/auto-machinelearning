// <copyright file="SweepableNode.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Sweeper;

namespace MLNet.AutoPipeline
{
    public class SweepableNode<TTransformer, TOption> : ISweepablePipelineNode
        where TTransformer : ITransformer
        where TOption : class
    {
        private readonly OptionBuilder<TOption> _optionBuilder;
        private readonly TransformerScope _scope;
        private readonly Func<TOption, IEstimator<TTransformer>> _estimatorFactory;

        public SweepableNode(Func<TOption, IEstimator<TTransformer>> estimatorFactory, OptionBuilder<TOption> optionBuilder, TransformerScope scope = TransformerScope.Everything)
        {
            this._estimatorFactory = estimatorFactory;
            this._optionBuilder = optionBuilder;
            this._scope = scope;
            this.ValueGenerators = optionBuilder.ValueGenerators;
            this.EstimatorName = estimatorFactory(optionBuilder.CreateDefaultOption()).ToString().Split('.').Last();
        }

        public string EstimatorName { get; private set; }

        public TransformerScope Scope => this._scope;

        public IValueGenerator[] ValueGenerators { get; private set; }

        public SweepablePipelineNodeType NodeType => SweepablePipelineNodeType.Sweepable;

        public IEstimator<ITransformer> BuildEstimator(ParameterSet parameters)
        {
            var option = this._optionBuilder.BuildOption(parameters);
            return this._estimatorFactory(option) as IEstimator<ITransformer>;
        }

        public string Summary()
        {
            return $"SweepableNode({this.EstimatorName})";
        }

        public override string ToString()
        {
            return this.Summary();
        }
    }
}
