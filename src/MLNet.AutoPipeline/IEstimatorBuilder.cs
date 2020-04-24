// <copyright file="IEstimatorBuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Sweeper;

namespace MLNet.AutoPipeline
{
    public interface IEstimatorBuilder
    {
        IEstimator<ITransformer> BuildEstimator(ParameterSet parameters);

        TransformerScope Scope { get; }

        string EstimatorName { get; }
    }

    public class EstimatorBuilder<TTransformer, TOption> : IEstimatorBuilder
        where TTransformer : ITransformer
        where TOption : class
    {
        private readonly OptionBuilder<TOption> _optionBuilder;
        private readonly TransformerScope _scope;
        private readonly Func<TOption, IEstimator<TTransformer>> _estimatorFactory;

        public EstimatorBuilder(Func<TOption, IEstimator<TTransformer>> estimatorFactory, OptionBuilder<TOption> optionBuilder, TransformerScope scope = TransformerScope.Everything)
        {
            this._estimatorFactory = estimatorFactory;
            this._optionBuilder = optionBuilder;
            this._scope = scope;
        }

        public string EstimatorName => $"{nameof(TTransformer)}, {nameof(TOption)}";

        public TransformerScope Scope => this._scope;

        public IEstimator<ITransformer> BuildEstimator(ParameterSet parameters)
        {
            var option = this._optionBuilder.BuildOption(parameters);
            return this._estimatorFactory(option) as IEstimator<ITransformer>;
        }
    }

    public class EstimatorWrapper<TTransformer> : IEstimatorBuilder
        where TTransformer : ITransformer
    {
        private IEstimator<TTransformer> _instance;
        private TransformerScope _scope;

        public EstimatorWrapper(IEstimator<TTransformer> instance, TransformerScope scope = TransformerScope.Everything)
        {
            this._instance = instance;
            this._scope = scope;
        }

        public ParameterSet Current { get => null; }

        public string EstimatorName => $"{typeof(TTransformer).Name}";

        public TransformerScope Scope => this._scope;

        public IEstimator<ITransformer> BuildEstimator(ParameterSet parameters)
        {
            return this._instance as IEstimator<ITransformer>;
        }
    }
}
