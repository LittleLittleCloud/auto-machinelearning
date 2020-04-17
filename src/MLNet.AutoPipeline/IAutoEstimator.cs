﻿// <copyright file="IAutoEstimator.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using System;

namespace Microsoft.ML.AutoPipeline
{
    internal interface IAutoEstimator
    {
        ISweeper Sweeper { get; }
    }

    internal interface IAutoEstimator<out TTransformer> : IEstimator<TTransformer>, IAutoEstimator
        where TTransformer : ITransformer
    {
        IEstimator<TTransformer> ToEstimator();
    }

    internal class AutoEstimator<TTransformer, TOption> : IAutoEstimator<TTransformer>
        where TTransformer : ITransformer
        where TOption : class
    {
        private readonly ISweeper _sweeper;
        private readonly OptionBuilder<TOption> _optionBuilder;
        private readonly Func<TOption, IEstimator<TTransformer>> _estimatorFactory;
        private IEstimator<TTransformer> _current;

        public AutoEstimator(Func<TOption, IEstimator<TTransformer>> estimatorFactory, OptionBuilder<TOption> optionBuilder, ISweeper sweeper)
        {
            this._estimatorFactory = estimatorFactory;
            this._optionBuilder = optionBuilder;
            this._sweeper = sweeper;
        }

        public ISweeper Sweeper => this._sweeper;

        public TTransformer Fit(IDataView input)
        {
            var sweepOutput = this._sweeper.Current;
            var option = this._optionBuilder.BuildOption(sweepOutput);
            this._current = this._estimatorFactory(option);
            return this._current.Fit(input);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            var defaultOption = this._optionBuilder.CreateDefaultOption();
            var defaultEstimator = this._estimatorFactory(defaultOption);
            return defaultEstimator.GetOutputSchema(inputSchema);
        }

        public bool MoveNext()
        {
            return this._sweeper?.MoveNext() ?? false;
        }

        public void Reset()
        {
            this._sweeper?.Reset();
        }

        public IEstimator<TTransformer> ToEstimator()
        {
            return this._current;
        }
    }
}
