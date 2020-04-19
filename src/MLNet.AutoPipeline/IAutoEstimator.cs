// <copyright file="IAutoEstimator.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;
using Microsoft.ML;
using MLNet.Sweeper;

namespace MLNet.AutoPipeline
{
    internal interface IAutoEstimator
    {
        ParameterSet Current { get; set; }
    }

    internal interface IAutoEstimator<out TTransformer> : IEstimator<TTransformer>, IAutoEstimator
        where TTransformer : ITransformer
    {
        IEstimator<TTransformer> ToEstimator(ParameterSet parameters);
    }

    internal class AutoEstimator<TTransformer, TOption> : IAutoEstimator<TTransformer>
        where TTransformer : ITransformer
        where TOption : class
    {
        private readonly OptionBuilder<TOption> _optionBuilder;
        private readonly Func<TOption, IEstimator<TTransformer>> _estimatorFactory;

        public AutoEstimator(Func<TOption, IEstimator<TTransformer>> estimatorFactory, OptionBuilder<TOption> optionBuilder)
        {
            this._estimatorFactory = estimatorFactory;
            this._optionBuilder = optionBuilder;
        }

        public ParameterSet Current { get; set; }

        public TTransformer Fit(IDataView input)
        {
            return this.ToEstimator(this.Current).Fit(input);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            var defaultOption = this._optionBuilder.CreateDefaultOption();
            var defaultEstimator = this._estimatorFactory(defaultOption);
            return defaultEstimator.GetOutputSchema(inputSchema);
        }

        public IEstimator<TTransformer> ToEstimator(ParameterSet parameters)
        {
            var option = this._optionBuilder.BuildOption(parameters);
            return this._estimatorFactory(option);
        }
    }
}
