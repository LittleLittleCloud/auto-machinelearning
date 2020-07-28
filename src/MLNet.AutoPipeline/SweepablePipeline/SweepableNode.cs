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
    public class SweepableNode<TNewTrain, TOption> : INode<TNewTrain>
        where TNewTrain : IEstimator<ITransformer>
        where TOption : class
    {
        private readonly OptionBuilder<TOption> _optionBuilder;
        private readonly TransformerScope _scope;
        private readonly Func<TOption, TNewTrain> _estimatorFactory;

        public SweepableNode(Func<TOption, TNewTrain> estimatorFactory, OptionBuilder<TOption> optionBuilder, TransformerScope scope = TransformerScope.Everything, string estimatorName = null, string[] inputs = null, string[] outputs = null)
        {
            this._estimatorFactory = estimatorFactory;
            this._optionBuilder = optionBuilder;
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

        public string EstimatorName { get; private set; }

        public TransformerScope Scope => this._scope;

        public IValueGenerator[] ValueGenerators { get; private set; }

        public NodeType NodeType => NodeType.Sweepable;

        public string[] InputColumns { get; private set; }

        public string[] OutputColumns { get; private set; }

        public TNewTrain BuildEstimator(ParameterSet parameters)
        {
            var option = this._optionBuilder.BuildOption(parameters);
            return this._estimatorFactory(option);
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
            var selectedUnsweepableParams = this._optionBuilder.UnsweepableParameters;
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
