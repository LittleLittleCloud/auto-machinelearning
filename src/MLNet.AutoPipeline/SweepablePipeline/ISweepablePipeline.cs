// <copyright file="ISweepablePipeline.cs" company="BigMiao">
// Copyright (c) BigMiao. All rights reserved.
// </copyright>

using Microsoft.ML;
using Microsoft.ML.Data;
using MLNet.Sweeper;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNet.AutoPipeline
{
    public interface ISweepablePipeline
    {
        IList<IValueGenerator> ValueGenerators { get; }

        IList<INode> SweepablePipelineNodes { get; }

        ISweeper Sweeper { get; }

        ISweepablePipeline Append(INode builder);

        ISweepablePipeline Append<TTransformer>(IEstimator<TTransformer> estimator, TransformerScope scope = TransformerScope.Everything)
            where TTransformer : ITransformer;

        ISweepablePipeline Append<TNewTran, TOption>(Func<TOption, IEstimator<TNewTran>> estimatorBuilder, OptionBuilder<TOption> optionBuilder, TransformerScope scope = TransformerScope.Everything)
            where TNewTran : ITransformer
            where TOption : class;

        ISweepablePipeline Concat(ISweepablePipeline chain);

        string Summary();

        void UseSweeper(ISweeper sweeper);

        IEnumerable<SweepingInfo> Sweeping(int maximum);
    }
}
