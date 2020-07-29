// <copyright file="SweepableNode.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Sweeper;

namespace MLNet.AutoPipeline
{
    public class SweepableNode<TTrain, TOption> : ISweepableNode<TTrain, TOption>
        where TTrain : IEstimator<ITransformer>
        where TOption : class
    {
        private readonly TransformerScope _scope;

        public SweepableNode(Func<TOption, TTrain> estimatorFactory, OptionBuilder<TOption> optionBuilder, TransformerScope scope = TransformerScope.Everything, string estimatorName = null, string[] inputs = null, string[] outputs = null)
        {
            this.EstimatorFactory = estimatorFactory;
            this.OptionBuilder = optionBuilder;
            this._scope = scope;
            this.ValueGenerators = optionBuilder.ValueGenerators;

            if (estimatorName == null)
            {
                this.EstimatorName = estimatorFactory(optionBuilder.CreateDefaultOption()).ToString().Split('.').Last();
            }
            else
            {
                this.EstimatorName = estimatorName;
            }

            this.InputColumns = inputs;
            this.OutputColumns = outputs;
        }

        public Func<TOption, TTrain> EstimatorFactory { get; private set; }

        public OptionBuilder<TOption> OptionBuilder { get; private set; }

        public string EstimatorName { get; private set; }

        public TransformerScope Scope => this._scope;

        public IValueGenerator[] ValueGenerators { get; private set; }

        public NodeType NodeType => NodeType.Sweepable;

        public string[] InputColumns { get; private set; }

        public string[] OutputColumns { get; private set; }

        public TTrain BuildEstimator(ParameterSet parameters)
        {
            var option = this.OptionBuilder.BuildOption(parameters);
            return this.EstimatorFactory(option);
        }

        public string Summary()
        {
            return $"SweepableNode({this.EstimatorName})";
        }

        public override string ToString()
        {
            return this.Summary();
        }

        internal CodeGenNodeContract ToCodeGenNodeContract(ParameterSet parameters)
        {
            var valueGeneratorIds = this.ValueGenerators.Select(x => x.ID).ToImmutableHashSet();
            var selectedParams = parameters.Where(x => valueGeneratorIds.Contains(x.ID));
            var selectedUnsweepableParams = this.OptionBuilder.UnsweepableParameters;
            return new CodeGenNodeContract()
            {
                EstimatorName = this.EstimatorName,
                InputColumns = this.InputColumns ?? (new string[] { }),
                OutputColumns = this.OutputColumns ?? new string[] { },
                Parameters = new ParameterSet(selectedParams.Concat(selectedUnsweepableParams)),
            };
        }

        IEstimator<ITransformer> INode.BuildEstimator(ParameterSet parameters)
        {
            return this.BuildEstimator(parameters) as IEstimator<ITransformer>;
        }
    }
}
