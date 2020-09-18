// <copyright file="SweepableEstimatorBase.cs" company="BigMiao">
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
    /// <summary>
    /// Abstract class for estimators in <see cref="SweepablePipeline"/> and <see cref="SingleEstimatorSweepablePipeline"/>.
    /// </summary>
    /// <typeparam name="TTran">transformer.</typeparam>
    public abstract class SweepableEstimatorBase : ISweepable<IEstimator<ITransformer>>, ISweepableEstimator
    {
        public SweepableEstimatorBase(string estimatorName, string[] inputColumns, string[] outputColumns, IEnumerable<IValueGenerator> sweepableValueGenerators, TransformerScope scope = TransformerScope.Everything)
        {
            this.EstimatorName = estimatorName;
            this.InputColumns = inputColumns;
            this.OutputColumns = outputColumns;
            this.SweepableValueGenerators = sweepableValueGenerators;
            this.Scope = scope;
        }

        public string EstimatorName { get; protected set; }

        // TODO
        // use dictionary instead of string array.
        public string[] InputColumns { get; protected set; }

        // TODO
        // use dictionary instead of string array.
        public string[] OutputColumns { get; protected set; }

        public TransformerScope Scope { get; protected set; }

        public IEnumerable<IValueGenerator> SweepableValueGenerators { get; protected set; }

        public override string ToString()
        {
            var sb = new StringBuilder();
            sb.Append(this.EstimatorName);

            if (this.InputColumns != null)
            {
                sb.Append($"(input_columns:[{string.Join(",", this.InputColumns)}])");
            }

            if (this.OutputColumns != null)
            {
                sb.Append($"(output_columns:[{string.Join(",", this.OutputColumns)}])");
            }

            return sb.ToString();
        }

        public abstract IEstimator<ITransformer> BuildFromParameters(IDictionary<string, string> parameters);
    }
}
