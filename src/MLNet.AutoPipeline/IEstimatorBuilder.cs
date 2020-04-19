// <copyright file="IEstimatorBuilder.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using Microsoft.ML;
using MLNet.Sweeper;

namespace MLNet.AutoPipeline
{
    public interface IEstimatorBuilder
    {
        IEstimator<ITransformer> BuildEstimator(ParameterSet parameters);
    }

    public class EstimatorBuilder<TTransformer, TOption> : IEstimatorBuilder
        where TTransformer : ITransformer
        where TOption : class
    {
        private readonly OptionBuilder<TOption> _optionBuilder;
        private readonly Func<TOption, IEstimator<TTransformer>> _estimatorFactory;

        public EstimatorBuilder(Func<TOption, IEstimator<TTransformer>> estimatorFactory, OptionBuilder<TOption> optionBuilder)
        {
            this._estimatorFactory = estimatorFactory;
            this._optionBuilder = optionBuilder;
        }

        public IEstimator<ITransformer> BuildEstimator(ParameterSet parameters)
        {
            var option = this._optionBuilder.BuildOption(parameters);
            return this._estimatorFactory(option) as IEstimator<ITransformer>;
        }
    }

    public class EstimatorWrapper : IEstimatorBuilder
    {
        private IEstimator<ITransformer> _instance;

        public EstimatorWrapper(IEstimator<ITransformer> instance)
        {
            this._instance = instance;
        }

        public ParameterSet Current { get => null; }

        public IEstimator<ITransformer> BuildEstimator(ParameterSet parameters)
        {
            return this._instance;
        }
    }
}
